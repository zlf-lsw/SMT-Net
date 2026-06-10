#!/usr/bin/env python3
"""
测试新数据集加载器的脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import torch
import yaml
from torch.utils.data import DataLoader

# 导入自定义模块
import data
import archs
import models

from basicsr.data import build_dataset, build_dataloader
from basicsr.utils.options import dict2str

def test_dataset():
    """测试数据集加载"""
    print("=== 测试数据集加载 ===")
    
    # 读取配置文件
    config_path = 'paper_options/UNet_climate_baseline.yml'
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    
    print("配置文件加载成功")
    print(f"训练数据集配置: {opt['datasets']['train']}")
    
    # 创建训练数据集
    try:
        train_dataset = build_dataset(opt['datasets']['train'])
        print(f"训练数据集创建成功，样本数量: {len(train_dataset)}")
        
        # 测试第一个样本
        sample = train_dataset[0]
        print(f"样本数据结构:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, dtype: {value.dtype}")
            else:
                print(f"  {key}: {value}")
                
        # 创建数据加载器
        train_loader = build_dataloader(
            train_dataset, 
            opt['datasets']['train'], 
            num_gpu=1, 
            dist=False
        )
        print(f"数据加载器创建成功，批次数量: {len(train_loader)}")
        
        # 测试一个批次
        for i, batch in enumerate(train_loader):
            print(f"批次 {i}:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}, dtype: {value.dtype}")
                else:
                    print(f"  {key}: {len(value) if isinstance(value, list) else value}")
            break
            
    except Exception as e:
        print(f"数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试验证数据集
    try:
        val_dataset = build_dataset(opt['datasets']['val'])
        print(f"验证数据集创建成功，样本数量: {len(val_dataset)}")
        
    except Exception as e:
        print(f"验证数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("数据集测试完成！")
    return True

def test_network():
    """测试网络架构"""
    print("\n=== 测试网络架构 ===")
    
    try:
        from basicsr.archs import build_network
        
        # 网络配置
        network_opt = {
            'type': 'UNet',
            'add_hgt': False,
            'upscale': 4,
            'num_in_ch': 7,
            'num_out_ch': 1,
            'activation': 'none'
        }
        
        # 创建网络
        net = build_network(network_opt)
        print(f"网络创建成功: {type(net)}")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        
        # 创建测试输入
        batch_size = 2
        test_input = {
            'lq': torch.randn(batch_size, 7, 16, 16).to(device)
        }
        
        with torch.no_grad():
            output = net(test_input)
            print(f"网络输出形状: {output.shape}")
            print(f"预期输出形状: [{batch_size}, 1, 64, 64]")
            
            if output.shape == (batch_size, 1, 64, 64):
                print("网络前向传播测试成功！")
                return True
            else:
                print("网络输出形状不匹配！")
                return False
                
    except Exception as e:
        print(f"网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("开始测试数据集和网络...")
    
    dataset_ok = test_dataset()
    network_ok = test_network()
    
    if dataset_ok and network_ok:
        print("\n✅ 所有测试通过！可以开始训练了。")
        print("\n开始训练命令:")
        print("python train.py -opt paper_options/UNet_climate_baseline.yml")
    else:
        print("\n❌ 测试失败，请检查错误信息。")
