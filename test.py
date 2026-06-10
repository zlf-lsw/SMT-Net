import os.path as osp
import os

import archs  # noqa: F401
import data  # noqa: F401
import models  # noqa: F401
import metrics
import logging
import torch
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        phase_lower = str(phase).lower()
        declared_phase = str(dataset_opt.get('phase', '')).lower()
        # 兼容键名非 test，但配置声明为 val/test 的数据集
        if ('test' not in phase_lower) and (declared_phase not in ['val', 'test']):
            continue
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    if opt['path'].get('pretrain_network_g', None) is None:
        opt['path']['pretrain_network_g'] = osp.join('experiments', opt['name'], 'models/net_g_latest.pth')
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        # current_iter 需为整数，以用于日志与文件名格式化
        model.validation(test_loader, current_iter=0, tb_logger=None, save_img=opt['val']['save_img'])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    # 确保从任意工作目录启动时，配置等相对路径以脚本目录为基准
    try:
        os.chdir(root_path)
    except Exception:
        pass
    test_pipeline(root_path)
