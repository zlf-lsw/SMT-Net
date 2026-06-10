import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import get_root_logger


@DATASET_REGISTRY.register()
class ClimateDataset(Dataset):
    """
    简化版数据集：仅支持当前 .pt 旧格式数据（张量内存格式），并与目录
    /home/lyb/diffusion/main2 copy/dataset 的实际文件对齐：

    期望 .pt 内包含 3 个键：
    - 'LR_input': [C, T, 16, 16]
    - 'HR_target': [C, T, 64, 64]
    - 'HR_topo'  : [2, 64, 64]

    数据将被重排为 [T, C, H, W]，并在 __getitem__ 时做 Z-score 标准化。
    可选：将 HR_topo 下采样到 16x16 并拼到 lq 作为额外通道（use_topo=true）。
    仅保留单个 HR 目标通道，由 target_channel_index 指定（默认 0）。
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.logger = get_root_logger()

        # 基础路径与划分
        self.data_root = opt.get('dataroot_gt', '../dataset')
        self.phase = str(opt.get('phase', 'train')).lower()
        force_split = opt.get('force_split', None)

        if force_split is not None:
            split = str(force_split).lower()
        else:
            split = 'train' if self.phase == 'train' else ('val' if self.phase == 'val' else 'test')

        filename = {
            'train': 'dict_s2s_train.pt',
            'val': 'dict_s2s_val.pt',
            'test': 'dict_s2s_test.pt'
        }[split]
        data_file = os.path.join(self.data_root, filename)
        if not os.path.isfile(data_file):
            # 兼容无独立 train 文件时的旧命名
            if split == 'train':
                alt = os.path.join(self.data_root, 'dict_s2s.pt')
                if os.path.isfile(alt):
                    data_file = alt
                else:
                    raise FileNotFoundError(f'Cannot find dataset file: {data_file}')
            else:
                raise FileNotFoundError(f'Cannot find dataset file: {data_file}')

        self.logger.info(f'Loading dataset: {data_file}')
        data = torch.load(data_file, map_location='cpu')

        if not (isinstance(data, dict) and 'LR_input' in data and 'HR_target' in data and 'HR_topo' in data):
            raise KeyError('Expected dict with keys LR_input, HR_target, HR_topo in .pt file')

        # 原始张量
        self.lr_data = data['LR_input']   # [C, T, 16, 16]
        self.hr_data = data['HR_target']  # [C, T, 64, 64]
        self.topo_data = data['HR_topo']  # [2, 64, 64]

        # 重排为 [T, C, H, W]
        if self.lr_data.dim() == 4 and self.lr_data.shape[0] <= 10 and self.lr_data.shape[1] >= 1000:
            self.lr_data = self.lr_data.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError('LR_input must be [C, T, H, W] with small C and large T')

        if self.hr_data.dim() == 4 and self.hr_data.shape[0] <= 10 and self.hr_data.shape[1] >= 1000:
            self.hr_data = self.hr_data.permute(1, 0, 2, 3).contiguous()
        else:
            raise ValueError('HR_target must be [C, T, H, W] with small C and large T')

        # 仅保留单个 HR 通道
        self.target_channel_index = int(self.opt.get('target_channel_index', 0))
        if not (0 <= self.target_channel_index < self.hr_data.size(1)):
            raise IndexError(f'target_channel_index={self.target_channel_index} out of range for HR channels={self.hr_data.size(1)}')
        self.hr_data = self.hr_data[:, self.target_channel_index:self.target_channel_index+1, :, :]

        self.logger.info(f'Unified LR shape [N,C,H,W]: {tuple(self.lr_data.shape)}')
        self.logger.info(f'Unified HR shape [N,1,H,W]: {tuple(self.hr_data.shape)}')

        # 统计量（float32）
        self._compute_statistics()

    def _compute_statistics(self):
        lr = self.lr_data.float()
        hr = self.hr_data.float()
        self.lr_mean = lr.mean(dim=(0, 2, 3)).view(-1, 1, 1)
        self.lr_std = lr.std(dim=(0, 2, 3)).view(-1, 1, 1)
        self.hr_mean = hr.mean(dim=(0, 2, 3)).view(-1, 1, 1)
        self.hr_std = hr.std(dim=(0, 2, 3)).view(-1, 1, 1)

        logger = self.logger
        logger.info(f'LR mean: {self.lr_mean.squeeze()}')
        logger.info(f'LR std: {self.lr_std.squeeze()}')
        logger.info(f'HR mean: {self.hr_mean.squeeze()}')
        logger.info(f'HR std: {self.hr_std.squeeze()}')

    def __len__(self):
        return int(self.lr_data.size(0))

    def _zscore(self, x, mean, std):
        return (x - mean) / (std + 1e-8)

    def __getitem__(self, index):
        lr = self.lr_data[index].float()
        hr = self.hr_data[index].float()

        # 标准化
        if self.opt.get('normalize', True):
            lr = self._zscore(lr, self.lr_mean, self.lr_std).float()
            hr = self._zscore(hr, self.hr_mean, self.hr_std).float()

        # 选择/重排输入通道
        total_in_ch = int(lr.size(0))
        ch_indices = self.opt.get('input_channel_indices', None)
        if ch_indices is not None:
            if any([(i < 0) or (i >= total_in_ch) for i in ch_indices]):
                raise IndexError(f'input_channel_indices {ch_indices} out of range for LR channels={total_in_ch}')
            lr = lr[ch_indices]
        else:
            input_channels = int(self.opt.get('input_channels', total_in_ch))
            if input_channels < total_in_ch:
                lr = lr[:input_channels]
            elif input_channels == 1 and total_in_ch > 1:
                lr = lr.mean(dim=0, keepdim=True)

        # 拼接 topo（可选）
        if self.opt.get('use_topo', False):
            lr_h, lr_w = int(self.lr_data.size(-2)), int(self.lr_data.size(-1))
            topo_lr = F.interpolate(self.topo_data.unsqueeze(0).float(), size=(lr_h, lr_w), mode='bilinear', align_corners=False).squeeze(0)
            lr = torch.cat([lr, topo_lr], dim=0)

        return {
            'lq': lr,
            'gt': hr,
            'lq_path': f'sample_{index}',
            'gt_path': f'sample_{index}',
            'info': [f'sample_{index}']
        }


@DATASET_REGISTRY.register()
class ClimateDatasetWithTopo(ClimateDataset):
    """包含地形输出（供 add_hgt 模型使用）"""

    def __getitem__(self, index):
        result = super().__getitem__(index)
        result['hgt'] = self.topo_data.float()
        return result


