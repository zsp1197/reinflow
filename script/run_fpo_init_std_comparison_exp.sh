#!/bin/bash

# 本脚本用于比较 FPO Init Std 在训练阶段使用不同初始噪声分布策略的效果。
# 固定 std_min=0.5, std_max=1.5
# 限制单个实验总时长不超过 8h

STD_MIN=0.5
STD_MAX=1.5
TIMEOUT_DURATION="8h"

echo "Starting FPO Init Std comparison experiments (Sync vs No_Sync)..."

# Experiment 1: train_std_sync=false (Training uses fixed N(0,1), inference uses adaptive)
echo "================================================================"
echo "Starting experiment 1/2: train_std_sync=false"
echo "Timeout set to ${TIMEOUT_DURATION}"
echo "================================================================"

cat <<EOF > notes.txt
[FPO Init Std Experiment - NO SYNC]
Environment: ant-medium-expert-v2 (Gym)
Policy: ft_fpo_init_std_mlp
std_min: ${STD_MIN}
std_max: ${STD_MAX}
train_std_sync: false (Training initial CFM noise is fixed N(0,1), inference is adaptive N(0, init_std))
EOF

timeout ${TIMEOUT_DURATION} python script/run.py \
    --config-dir=cfg/gym/finetune/ant-v2 \
    --config-name=ft_fpo_init_std_mlp \
    run_name="fpo_init_std_nosync_min${STD_MIN}_max${STD_MAX}" \
    model.std_min=${STD_MIN} \
    model.std_max=${STD_MAX} \
    model.train_std_sync=false \
    device=cuda:0 \
    sim_device=cuda:0 \
    wandb=online

# `$?` is 124 if `timeout` command terminates the script
if [ $? -eq 124 ]; then
    echo "Experiment 1 timed out after ${TIMEOUT_DURATION}. Moving to next..."
else
    echo "Experiment 1 finished before timeout."
fi


# Experiment 2: train_std_sync=true (Training uses adaptive noise, synchronized with inference)
echo "================================================================"
echo "Starting experiment 2/2: train_std_sync=true"
echo "Timeout set to ${TIMEOUT_DURATION}"
echo "================================================================"

cat <<EOF > notes.txt
[FPO Init Std Experiment - SYNC]
Environment: ant-medium-expert-v2 (Gym)
Policy: ft_fpo_init_std_mlp
std_min: ${STD_MIN}
std_max: ${STD_MAX}
train_std_sync: true (Training initial CFM noise uses adaptive N(0, init_std), identical to inference)
EOF

timeout ${TIMEOUT_DURATION} python script/run.py \
    --config-dir=cfg/gym/finetune/ant-v2 \
    --config-name=ft_fpo_init_std_mlp \
    run_name="fpo_init_std_sync_min${STD_MIN}_max${STD_MAX}" \
    model.std_min=${STD_MIN} \
    model.std_max=${STD_MAX} \
    model.train_std_sync=true \
    device=cuda:0 \
    sim_device=cuda:0 \
    wandb=online

if [ $? -eq 124 ]; then
    echo "Experiment 2 timed out after ${TIMEOUT_DURATION}."
else
    echo "Experiment 2 finished before timeout."
fi

echo "================================================================"
echo "All comparisons completed."
