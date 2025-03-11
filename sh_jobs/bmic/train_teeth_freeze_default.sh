#!/bin/bash
#SBATCH --job-name=crop_seg_hiera_conv_resume
#SBATCH --output=sbatch_log/crop_seg_hiera_conv_resume_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=octopus04
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qi.ma@vision.ee.ethz.ch

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate sam2_cbct
export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOM

cd /srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/sam2_cbct/ 

export TORCH_USE_CUDA_DSA=1
export PYTHONPATH=./
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_teeth_finetune_default.yaml \
    --use-cluster 0 \
    --num-gpus 1 \

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