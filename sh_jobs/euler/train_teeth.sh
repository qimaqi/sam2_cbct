#!/bin/bash
#SBATCH --job-name=sam2                        # Job name
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks=8    
#SBATCH --ntasks-per-node=8                    # Number of tasks (GPUs) per node
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00                        # Max time (HH:MM:SS)
#SBATCH --output=sbatch_log/teeth_freeze_encoder_%j.log                 # Output file
#SBATCH --gpus=rtx_3090:8

module load eth_proxy
module load stack/2024-06
module load cuda/12.1.1
module load gcc/12.2.0
module load cmake/3.27.7
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda deactivate
conda activate sam2
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_teeth_finetune_default.yaml \
    --use-cluster 0 \
    --num-gpus 8 \

# debug resume



# python training/train.py \
#     -c configs/sam2.1_training/sam2.1_hiera_t_MOSE_finetune_teeth.yaml  \
#     --use-cluster 0 \
#     --num-gpus 2 \
       
        # -c configs/sam2.1_training/sam2.1_hiera_t_MOSE_finetune_teeth.yaml \
    
# sbatch --output=sbatch_log/debug_%j.out  --ntasks=8 --mem-per-cpu=4g --gpus=titan_rtx:2 --time=4-0 train.sh
# srun --ntasks=8 --mem-per-cpu=4G --gpus=rtx_3090:2  --time=240 --pty bash -i
# # srun --ntasks=8 --mem-per-cpu=4G  --time=240 --pty bash -i
# 12839180

# check multi gpu, batch size will be shared or not, no each dataloader is independent

# srun --ntasks=8 --mem-per-cpu=4G  --time=240 --gpus=rtx_3090:2 --pty bash -i
# check base model freezed can be used for 3090 or not