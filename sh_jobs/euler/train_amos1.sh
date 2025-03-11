#!/bin/bash
#SBATCH --job-name=sam2                        # Job name
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks=8    
#SBATCH --ntasks-per-node=8                    # Number of tasks (GPUs) per node
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00                        # Max time (HH:MM:SS)
#SBATCH --output=sbatch_log/amos_nfreeze_8gpu_point_%j.log                 # Output file
#SBATCH --gpus=rtx_3090:8


module load eth_proxy
module load stack/2024-06
module load cuda/12.1.1
module load gcc/12.2.0
module load cmake/3.27.7
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda deactivate
conda activate sam2
# conda create -n sam2 python=3.10
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


# python training/train.py \
#     -c configs/sam2.1_training/sam2.1_hiera_t_MOSE_finetune_amos_freeze.yaml \
#     --use-cluster 0 \
#     --num-gpus 8 \

# python training/train.py \
#     -c configs/sam2.1_training/sam2.1_hiera_t_MOSE_finetune_amos.yaml \
#     --use-cluster 0 \
#     --num-gpus 8 \


python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune_amos_point \
    --use-cluster 0 \
    --num-gpus 8 \


# python training/train.py \
#     -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune_amos_box \
#     --use-cluster 0 \
#     --num-gpus 1 \

