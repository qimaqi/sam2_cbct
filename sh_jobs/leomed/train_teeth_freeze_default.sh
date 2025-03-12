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

source /cluster/apps/bmicgroup/qimaqi/miniconda3/bin/activate
conda activate sam2_cbct

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOM

cd /cluster/customapps/bmicgroup/bmic/qimaqi/sam2_cbct

export PYTHONPATH=./
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_teeth_finetune_default_leomed.yaml \
    --use-cluster 0 \
    --num-gpus 1 \
