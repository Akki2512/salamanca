#!/bin/bash
cd /dev/shm/sonic/GR00T-WholeBodyControl
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HF_HOME=/dev/shm/sonic/hf TMPDIR=/dev/shm/sonic/tmp WANDB_MODE=disabled HOME=/dev/shm/sonic/home PYTORCH_JIT=0
../venv/bin/python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release +checkpoint=sonic_release/last.pt num_envs=4096 headless=True ++algo.config.num_learning_iterations=1500 ++callbacks.model_save.save_frequency=50 ++manager_env.commands.motion.motion_lib_cfg.motion_file=/dev/shm/sonic/data/crawl/robot ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy use_wandb=false > /dev/shm/sonic/finetune1.log 2>&1
echo EXIT $? >> /dev/shm/sonic/finetune1.log
