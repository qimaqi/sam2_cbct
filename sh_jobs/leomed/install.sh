#!/bin/bash

conda create -n sam2_cbct python=3.10
conda activate sam2_cbct
conda install -c nvidia/label/cuda-11.8.0 cuda=11.8 cuda-toolkit=11.8 cuda-nvcc=11.8
export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH 
# important for nvcc compiling process

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 tensordict -c pytorch -c nvidia -c conda-forge 
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
git remote add upstream https://github.com/facebookresearch/sam2.git                            
git fetch upstream
git merge upstream/main


TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9" SAM2_BUILD_ALLOW_ERRORS=0 pip install -e ".[notebooks]"
#   The detected CUDA version (11.8) mismatches the version that was used to compile
#   PyTorch (12.4). Please make sure to use the same CUDA versions.
    

# need to specify the cuda arch list for the compilation of the cuda kernels
pip install SimpleITK scikit-image scikit-learn opencv-python tensorboard fvcore pandas submitit
# pip install submitit
# pip install tensorboard
# pip install fvcore
# pip install pandas
