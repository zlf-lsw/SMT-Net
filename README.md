# SMT-Net  Climate Data Spatial Downscaling Framework
A deep learning framework for climate data spatial downscaling based on Transformer and Mamba architectures, supporting multiple state-of-the-art image super-resolution architectures.

1、Overview
This project implements spatial downscaling of climate data from low resolution (16×16) to high resolution (64×64), with a scale factor of 4x. The framework is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR) and supports various deep learning architectures for fine-grained prediction of climate variables.

| Model | Description | Config File |
|-------|-------------|-------------|
| **SwinIR + VMamba + HGT** | Hybrid SwinIR and VMamba architecture with topographic information | `SwinIR_VMamba_hgt.yml` |
| **SwinIR + VMamba** | Hybrid SwinIR and VMamba architecture | `SwinIR_VMamba.yml` |
| **SwinIR + HGT** | Swin Transformer-based architecture with topographic fusion | `SwinIR_climate_baseline_hgt.yml` |
| **SwinIR** | Swin Transformer-based image restoration architecture | `SwinIR_climate_baseline.yml` |
| **Uformer** | Transformer-based U-Net architecture | `Uformer_climate_baseline.yml` |
| **UNet** | Classic U-Net convolutional neural network | `UNet_climate_baseline.yml` |
| **SRCNN** | Simple super-resolution CNN | `SRCNN_climate_baseline.yml` |

2、Directory Structure

├── archs/                      # Network architecture definitions

│   ├── climate_swinir_arch.py  # SwinIR climate architecture

│   ├── climate_vmamba_arch.py  # VMamba climate architecture

│   ├── climate_vmamba_hybrid_arch.py  # SwinIR+VMamba hybrid architecture

│   ├── climate_unet_arch.py    # UNet architecture

│   ├── climate_uformer_arch.py # Uformer architecture

│   ├── climate_srcnn_arch.py   # SRCNN architecture

│   ├── climate_rcan_arch.py    # RCAN architecture

│   └── utils.py                # Common utilities (HGTNet, etc.)

├── data/                       # Dataset definitions

│   ├── climate_dataset.py      # Climate dataset loader

│   ├── transforms.py           # Data augmentation

│   └── merge_dataset.py        # Dataset merging utility

├── models/                     # Model definitions

│   ├── simple_climate_model.py # Simple climate model

│   ├── climatesr_model.py      # Climate super-resolution model

│   └── climateswinir_model.py  # SwinIR climate model

├── metrics/                    # Evaluation metrics

│   ├── calculate_climate_mae_mse.py    # MAE/MSE calculation

│   ├── calculate_climate_psnr_ssim.py  # PSNR/SSIM calculation

│   └── calculate_climate_advanced.py   # Advanced metrics (ACC, PSD, etc.)

├── paper_options/              # Configuration files

├── experiments/                # Experiment output directory

├── tools/                      # Visualization tools

├── train.py                    # Training script

└── test.py                     # Testing script

3、Environment Setup
conda create -n climate_ds python=3.9
conda activate climate_ds
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install basicsr
pip install numpy scipy matplotlib opencv-python einops timm
pip install mamba-ssm

4、Dataset Preparation
The dataset is in `.pt` format (PyTorch tensor files), containing the following keys:
- `LR_input`: Low-resolution input, shape `[C, T, 16, 16]` (C=channels, T=time steps)
- `HR_target`: High-resolution target, shape `[C, T, 64, 64]`
- `HR_topo`: Topographic data, shape `[2, 64, 64]` (elevation and slope)

dataset/

├── dict_s2s_train.pt    # Training set

├── dict_s2s_val.pt      # Validation set

└── dict_s2s_test.pt     # Test set


5、Training
### Single GPU Training
python train.py -opt paper_options/SwinIR_VMamba_hgt.yml

### Multi-GPU Distributed Training
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py -opt paper_options/SwinIR_VMamba_hgt.yml --launcher pytorch
    
**Run Testing**
python test.py -opt paper_options/SwinIR_VMamba_hgt_infer.yml

6、Experiment Results

Training results are saved in the `experiments/` directory:

```
experiments/<experiment_name>/
├── models/                 # Model checkpoints
│   ├── net_g_latest.pth   # Latest model
│   └── net_g_XXXXX.pth    # Model at specific iteration
├── training_states/        # Training states
├── visualization/          # Visualization results
└── *.log                   # Training logs



