#!/bin/bash
#SBATCH --job-name=unzip
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/eval_%j.log      # Redirect stdout to a log file
#SBATCH --time=24:00:00


module load eth_proxy
module load stack/2024-06
module load cuda/12.1.1
module load gcc/12.2.0
module load cmake/3.27.7
source /cluster/work/cvl/qimaqi/miniconda3/etc/profile.d/conda.sh 
conda deactivate
conda activate cbct
# conda create -n sam2 python=3.10
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
export CUDA_HOME=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python custom_compute_metrics.py --eval_dir=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs_video/amos_finetune_freeze_image_cbct 

# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_finetune_freeze_image

# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_finetune_jointly