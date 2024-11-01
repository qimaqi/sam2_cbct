#!/bin/bash
#SBATCH --job-name=scannet++
#SBATCH --output=sbatch_log/gsplat_scannet++_0_100_%j.out


## srun --ntasks=8 --mem-per-cpu=4G --gpus=rtx_2080_ti:1  --time=240 --pty bash -i
## srun --ntasks=8 --mem-per-cpu=4G --gpus=titan_rtx:1  --time=240 --pty bash -i
module load eth_proxy
module load stack/2024-06
module load cuda/12.1.1
module load gcc/12.2.0
module load cmake/3.27.7
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda deactivate
# conda create -n sam2 python=3.10
# conda activate sam2
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

start_idx=0
end_idx=1

# python scannet_train_euler_list.py --start_idx=$start_idx --end_idx=$end_idx  --strategy=default
python train_3d.py -net sam2 -exp_name BTCV_MedSAM2 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv

