#!/bin/bash

# 本脚本用于测试 fpo_init_std 算法下，不同的 std_min 和 std_max 取值对训练效果的影响。
# 固定 inference_steps 和 ft_denoising_steps。

STD_MINS=(0.1 0.4)
STD_MAXS=(1.0 1.5)

echo "Starting FPO Init Std comparison experiments..."

for std_min in "${STD_MINS[@]}"; do
    for std_max in "${STD_MAXS[@]}"; do
        echo "================================================================"
        echo "Starting experiment with std_min=${std_min}, std_max=${std_max}"
        echo "================================================================"

        # 动态写入 notes.txt，增加关于自适应初始方差的备注
        cat <<EOF > notes.txt
[FPO Init Std Experiment]
Environment: ant-medium-expert-v2 (Gym)
Policy: ft_fpo_init_std_mlp (FPOInitStdFlow)
std_min: ${std_min}
std_max: ${std_max}

Objective: This experiment evaluates the impact of dynamic initialization standard deviation with minimum ${std_min} and maximum ${std_max}.
EOF

        # 开始执行当前参数下的训练配置
        python script/run.py \
            --config-dir=cfg/gym/finetune/ant-v2 \
            --config-name=ft_fpo_init_std_mlp \
            run_name="fpo_init_std_min${std_min}_max${std_max}" \
            model.std_min=${std_min} \
            model.std_max=${std_max} \
            device=cuda:0 \
            sim_device=cuda:0 \
            wandb=online

        echo "Finished experiment with std_min=${std_min}, std_max=${std_max}"
        echo "================================================================"
    done
done

echo "All FPO init_std experiments completed successfully!"
