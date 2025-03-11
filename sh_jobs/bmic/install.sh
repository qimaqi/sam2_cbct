#!/bin/bash

conda create -n sam2_cbct python=3.10
conda activate sam2_cbct
conda install -c nvidia/label/cuda-11.8.0 cuda=11.8 cuda-toolkit=11.8 cuda-nvcc=11.8
export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH # important for nvcc compiling process

# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 tensordict -c pytorch -c nvidia -c conda-forge 
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
git remote add upstream https://github.com/facebookresearch/sam2.git                            
git fetch upstream
git merge upstream/main

pip install -e ".[notebooks]"
# [rank0]: Traceback (most recent call last):
# [rank0]:   File "/scratch_net/schusch/qimaqi/miniconda3/envs/sam2_cbct/lib/python3.10/site-packages/hydra/_internal/instantiate/_instantiate2.py", line 92, in _call_target
# [rank0]:     return _target_(*args, **kwargs)
# [rank0]:   File "/srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/sam2_cbct/training/trainer.py", line 228, in __init__
# [rank0]:     self._setup_ddp_distributed_training(distributed, accelerator)
# [rank0]:   File "/srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/sam2_cbct/training/trainer.py", line 295, in _setup_ddp_distributed_training
# [rank0]:     self.model = nn.parallel.DistributedDataParallel(
# [rank0]:   File "/scratch_net/schusch/qimaqi/miniconda3/envs/sam2_cbct/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 825, in __init__
# [rank0]:     _verify_param_shape_across_processes(self.process_group, parameters)
# [rank0]:   File "/scratch_net/schusch/qimaqi/miniconda3/envs/sam2_cbct/lib/python3.10/site-packages/torch/distributed/utils.py", line 288, in _verify_param_shape_across_processes
# [rank0]:     return dist._verify_params_across_processes(process_group, tensors, logger)
# [rank0]: RuntimeError: CUDA error: named symbol not found
# [rank0]: CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# [rank0]: For debugging consider passing CUDA_LAUNCH_BLOCKING=1
# [rank0]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


# [rank0]: The above exception was the direct cause of the following exception:

# [rank0]: Traceback (most recent call last):
# [rank0]:   File "/srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/sam2_cbct/training/train.py", line 270, in <module>
# [rank0]:     main(args)
# [rank0]:   File "/srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/sam2_cbct/training/train.py", line 240, in main
# [rank0]:     single_node_runner(cfg, main_port)
# [rank0]:   File "/srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/sam2_cbct/training/train.py", line 53, in single_node_runner
# [rank0]:     single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
# [rank0]:   File "/srv/beegfs-benderdata/scratch/qimaqi_data/data/miccai_2025/sam2_cbct/training/train.py", line 40, in single_proc_run
# [rank0]:     trainer = instantiate(cfg.trainer, _recursive_=False)
# [rank0]:   File "/scratch_net/schusch/qimaqi/miniconda3/envs/sam2_cbct/lib/python3.10/site-packages/hydra/_internal/instantiate/_instantiate2.py", line 226, in instantiate
# [rank0]:     return instantiate_node(
# [rank0]:   File "/scratch_net/schusch/qimaqi/miniconda3/envs/sam2_cbct/lib/python3.10/site-packages/hydra/_internal/instantiate/_instantiate2.py", line 347, in instantiate_node
# [rank0]:     return _call_target(_target_, partial, args, kwargs, full_key)
# [rank0]:   File "/scratch_net/schusch/qimaqi/miniconda3/envs/sam2_cbct/lib/python3.10/site-packages/hydra/_internal/instantiate/_instantiate2.py", line 97, in _call_target
# [rank0]:     raise InstantiationException(msg) from e
# [rank0]: hydra.errors.InstantiationException: Error in call to target 'training.trainer.Trainer':
# [rank0]: RuntimeError('CUDA error: named symbol not found\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n')
# [rank0]: full_key: trainer

TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9" SAM2_BUILD_ALLOW_ERRORS=0 pip install -e ".[notebooks]"
#   The detected CUDA version (11.8) mismatches the version that was used to compile
#   PyTorch (12.4). Please make sure to use the same CUDA versions.
    

# need to specify the cuda arch list for the compilation of the cuda kernels
pip install SimpleITK scikit-image scikit-learn opencv-python tensorboard fvcore pandas submitit
# pip install submitit
# pip install tensorboard
# pip install fvcore
# pip install pandas
