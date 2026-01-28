#!/bin/bash

# 严格模式，遇到错误立即停止
set -e

# 项目配置
PROJECT_NAME="spt"
PYTHON_VERSION="3.8"
# 目标 CUDA 版本 (通过 Conda 安装，不影响系统 CUDA)
CUDA_VERSION_MAJOR="12"
CUDA_VERSION_MINOR="1"
CUDA_FULL_VERSION="12.1.1"  # 用于 Conda 包指定
TORCH_VERSION="2.2.0"

echo "_____________________________________________"
echo "   🧩 Superpoint Transformer (兼容模式) 🤖  "
echo "   系统 CUDA: 保持不变 (masked)"
echo "   环境 CUDA: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} (via Conda)"
echo "_____________________________________________"

# 获取脚本所在目录
HERE=$(dirname "$0")
HERE=$(realpath "$HERE")
cd "$HERE"

# 1. 初始化 Conda 环境
echo "⭐ [1/5] 正在创建 Conda 环境 '${PROJECT_NAME}'..."

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "Error: 找不到 conda 命令，请确保已安装 Conda。"
    exit 1
fi

# 如果环境已存在，询问是否删除 (为自动化脚本，这里我们尝试删除重建，或者用户需手动处理)
# 为了安全，如果存在则报错提示用户手动删除，或者尝试更新
if conda info --envs | grep -q "^${PROJECT_NAME} "; then
    echo "警告: 环境 '${PROJECT_NAME}' 已存在。"
    echo "正在移除旧环境以确保安装干净..."
    conda env remove -n ${PROJECT_NAME} -y
fi

# 创建环境并安装 Python
conda create -n ${PROJECT_NAME} python=${PYTHON_VERSION} -y

# 激活环境 (在脚本中需要使用 hook)
eval "$(conda shell.bash hook)"
conda activate ${PROJECT_NAME}

# 2. 安装 CUDA Toolkit (关键步骤：通过 Conda 提供 nvcc)
echo "⭐ [2/5] 正在安装 CUDA Toolkit ${CUDA_FULL_VERSION}..."
# 使用 nvidia channel 安装 cuda-toolkit (包含 nvcc)
conda install -n ${PROJECT_NAME} -c "nvidia/label/cuda-${CUDA_FULL_VERSION}" cuda-toolkit -y

# 设置环境变量，确保编译时使用 Conda 的 CUDA
export CUDA_HOME=${CONDA_PREFIX}
export PATH=${CONDA_PREFIX}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

echo "   验证环境内 CUDA 版本:"
nvcc --version

# 3. 安装 PyTorch
echo "⭐ [3/5] 正在安装 PyTorch ${TORCH_VERSION}..."
# 对应 CUDA 12.1 的 PyTorch
pip install torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}

# 4. 安装其他依赖
echo "⭐ [4/5] 正在安装 Python 依赖..."

# 基础工具
conda install pip nb_conda_kernels -y

# 数据科学与可视化
pip install matplotlib
pip install plotly==5.9.0
pip install "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash
pip install "notebook>=5.3" "ipywidgets>=7.5"
pip install ipykernel

# 机器学习工具
pip install torchmetrics==0.11.4

# PyG (PyTorch Geometric) - 需要匹配 Torch 和 CUDA 版本
echo "   安装 PyG 相关库..."
pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.html
pip install torch_geometric==2.3.0

# 项目特定依赖
pip install plyfile h5py colorhash seaborn numba
pip install pytorch-lightning
pip install pyrootutils
pip install hydra-core --upgrade
pip install hydra-colorlog
pip install hydra-submitit-launcher
pip install "rich<=14.0"
pip install torch_tb_profiler
pip install wandb
pip install open3d
pip install gdown
pip install ipyfilechooser

# 编译型依赖 (可能需要较长时间)
echo "   安装编译型扩展 (torch-ransac3d, pgeof 等)..."
pip install torch-ransac3d
pip install pgeof
pip install pycut-pursuit
pip install pygrid-graph

# 5. 安装 FRNN (从源码编译)
echo "⭐ [5/5] 正在安装 FRNN..."
mkdir -p src/dependencies

# 如果目录不存在则克隆
if [ ! -d "src/dependencies/FRNN" ]; then
    echo "   Cloning FRNN..."
    git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN
else
    echo "   FRNN 目录已存在，跳过克隆..."
fi

# 安装 prefix_sum
echo "   编译 prefix_sum..."
cd src/dependencies/FRNN/external/prefix_sum
pip install .

# 安装 FRNN
echo "   编译 FRNN..."
cd ../../ # 回到 FRNN 根目录
# 确保 setup.py 能找到 conda 的 nvcc
export CUDA_HOME=${CONDA_PREFIX}
pip install .

cd ../../../ # 回到项目根目录

echo "_____________________________________________"
echo "✅ 安装完成! 请使用以下命令激活环境:"
echo "   conda activate ${PROJECT_NAME}"
echo "_____________________________________________"
