#!/bin/bash
# usage: export.sh <checkpoint.pt>  -> writes ONNX files into exported/ next to the checkpoint
cd /dev/shm/sonic/GR00T-WholeBodyControl
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y HF_HOME=/dev/shm/sonic/hf TMPDIR=/dev/shm/sonic/tmp WANDB_MODE=disabled HOME=/dev/shm/sonic/home PYTORCH_JIT=0
../venv/bin/python gear_sonic/eval_agent_trl.py +checkpoint=$1 +headless=True ++num_envs=1 +export_onnx_only=true "++manager_env.commands.motion.motion_lib_cfg.motion_file=/dev/shm/sonic/data/crawl/robot" "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" > /dev/shm/sonic/export.log 2>&1
echo EXIT $? >> /dev/shm/sonic/export.log
