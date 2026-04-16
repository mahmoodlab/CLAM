#!/bin/bash

# 创建并配置 CLAM 特征提取环境

ENV_NAME="clam_extract"

echo "=== 创建 conda 环境: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.10 -y

echo "=== 激活环境 ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "=== 安装 PyTorch (CUDA 11.8) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "=== 安装基础依赖 ==="
pip install timm
pip install h5py
pip install openslide-python
pip install tqdm
pip install numpy
pip install Pillow
pip install pandas

echo "=== 安装 CONCH ==="
pip install git+https://github.com/Mahmoodlab/CONCH.git

echo "=== 安装 transformers (用于 CONCH v1.5) ==="
pip install transformers

echo "=== 完成 ==="
echo ""
echo "使用方法:"
echo "  conda activate $ENV_NAME"
echo "  export HF_AUTH_TOKEN='your_token'  # 可选"
echo "  torchrun --nproc_per_node=4 extract_features_ddp.py --model_name conch_v1 ..."
echo ""
echo "注意: 请确保系统已安装 openslide-tools"
echo "  Ubuntu/Debian: sudo apt-get install openslide-tools"
echo "  CentOS/RHEL: sudo yum install openslide"
