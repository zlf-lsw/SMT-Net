import os.path as osp
import os

import archs  # noqa: F401
import data  # noqa: F401
import models  # noqa: F401
import metrics # noqa: F401
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    # 确保从任意工作目录启动时，配置等相对路径以脚本目录为基准
    try:
        os.chdir(root_path)
    except Exception:
        pass
    train_pipeline(root_path)
