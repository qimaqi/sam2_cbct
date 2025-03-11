#!/bin/bash
#SBATCH --job-name=sam2                        # Job name
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks=1    
#SBATCH --ntasks-per-node=1                    # Number of tasks (GPUs) per node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G                      # Memory per processor
#SBATCH --time=24:00:00                        # Max time (HH:MM:SS)
#SBATCH --output=sbatch_log/infer_log_%j.log                 # Output file
#SBATCH --gpus=rtx_3090:1

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


python ./tools/vos_inference.py \
  --sam2_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint /cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_teeth_finetune_point_only.yaml/checkpoints/checkpoint.pt \
  --base_video_dir /cluster/work/cvl/qimaqi/miccai_2025/teeth_datasets/teeth_videos/val/JPEGImages \
  --input_mask_dir /cluster/work/cvl/qimaqi/miccai_2025/teeth_datasets/teeth_videos/val/Annotations \
  --output_mask_dir ./outputs/align_teeth_finetune_freeze_image_all_point \
  --use_all_masks \
  --track_object_appearing_later_in_video \
  --point_prompt 


python ./tools/vos_inference.py \
  --sam2_cfg  configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint /cluster/work/cvl/qimaqi/miccai_2025/sam2_cbct/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_teeth_finetune_point_only.yaml/checkpoints/checkpoint.pt \
  --base_video_dir /cluster/work/cvl/qimaqi/miccai_2025/teeth_datasets/teeth_videos/val/JPEGImages \
  --input_mask_dir /cluster/work/cvl/qimaqi/miccai_2025/teeth_datasets/teeth_videos/val/Annotations \
  --output_mask_dir ./outputs/align_teeth_finetune_freeze_image_track_point \
  --track_object_appearing_later_in_video \
  --point_prompt 