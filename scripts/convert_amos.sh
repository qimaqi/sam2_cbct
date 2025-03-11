#!/bin/bash
#SBATCH --job-name=unzip
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./joblogs/convert_%j.log      # Redirect stdout to a log file
#SBATCH --time=24:00:00


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

# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_freeze_pred

# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_finetune_freeze_image

# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_finetune_jointly


# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_finetune_jointly_point1_all

# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_finetune_jointly_track

# conda activate cbct
# python custom_compute_metrics.py --eval_dir=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs_video/amos_finetune_jointly_track_cbct


# python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/inputs/cbct

python Convert_Videos_to_CBCT.py --input_path=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs/amos_finetune_freeze_image_track_subset


conda activate cbct
python custom_compute_metrics.py --eval_dir=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs_video/amos_finetune_freeze_image_track_subset_cbct


# python custom_compute_metrics.py --eval_dir=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs_video/amos_freeze_pred_track_cbct


# python custom_compute_metrics.py --eval_dir=/cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/outputs_video/amos_finetune_jointly_point1_all_cbct
