1. Create ssh key pair
   ssh-keygen -t ed25519 -C "<>@gmail.com"
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   cat ~/.ssh/id_ed25519.pub
   copy and add to github account

2. git clone git@github.com:<>/open-r1.git
   git checkout dev

3. commands
    pip install uv
    uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip --link-mode=copy
    uv pip install vllm==0.7.2 --link-mode=copy

    uv pip install setuptools && uv pip install flash-attn --no-build-isolation --link-mode=copy
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]" --link-mode=copy --no-build-isolation

    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]" --link-mode=copy

    if anything fails, https://github.com/huggingface/open-r1/issues/282

    pip install tensorboard
    pip install wandb

    python -c "import torch; print(torch.__version__)"
    python -c 'import deepspeed; print("DeepSpeed is working!")'
    python -c 'import flash_attn; print("FlashAttention is working!")'

    huggingface-cli login
    wandb login

    git-lfs --version
    sudo apt-get install git-lfs

    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml

    cat ~/.cache/huggingface/token
    Add huggingface token to run_r1_grpo.py

    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 training/run_r1_grpo.py \
    --config training/grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml

    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file training/deepspeed_zero2.yaml \
    --num_processes=8 training/run_r1_grpo.py \
    --config training/grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml |& tee output.txt

    ======================== DEBUG ========================

   Ensure you really have 8 GPUs:
         1. confirm number of GPUs
            nvidia-smi 
         2. confirm CUDA version
            nvcc --version
         3. confirm GCC version
            gcc --version
   Monitor GPU usage:
      Use nvidia-smi to monitor GPU usage and see if any GPU is underutilized or has a high memory usage.
   Try with fewer GPUs:
      Temporarily set num_processes: 1 or num_processes: 2 and see if it still segfaults. That quickly tells you if the crash is from multi‑GPU synchronization or something else.
   Explicitly map devices:
      If you see “PG ID 0 ... using GPU 0 to perform barrier as devices used by this process are unknown,” it can help to set environment variables like:
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch ...
      and/or configure the device mapping in your Python code or accelerate config.
   Check if the same config works without vLLM:
      If everything is stable without use_vllm: true, then the fault might be an interaction between vLLM and DeepSpeed’s usage of GPU memory.
      Look for logs or OOM errors:
   Check the logs for any error messages.
      Sometimes a segfault is actually an out-of-memory error or driver crash. Check dmesg or /var/log/syslog for hints.
   Try    
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      upgrading GCC to 12
