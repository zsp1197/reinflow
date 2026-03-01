#!/bin/bash

# 本脚本用于在一张 4090 显卡上，串行按次序运行不同 inference step 的 PPOFlow 实验。
# 依次测试的 inference step (denoising_steps) 值为: 1, 3, 5, 7, 9

STEPS=(1 3 5 7 9)

echo "Starting sequential ReinFlow experiments for different inference steps..."

for step in "${STEPS[@]}"; do
    echo "================================================================"
    echo "Starting experiment with denoising_steps=${step}"
    echo "================================================================"

    # 动态写入 notes.txt，方便 wandb.init 时获取对应的环境、网络和超参数描述
    cat <<EOF > notes.txt
[Hyperparameter Search]
Environment: ant-medium-expert-v2 (Gym)
Policy: ft_ppo_reflow_mlp (PPOFlow)
Denoising Inference Steps: ${step}

This run compares the training effects across different inference steps (1, 3, 5, 7, 9).
EOF

    # 开始执行当前参数下的训练配置
    # 注意：在 ft_ppo_reflow_mlp.yaml 中，对应的控制变量是 `denoising_steps` 以及 `ft_denoising_steps`
    PYTHONPATH=$(pwd) python script/run.py \
        --config-dir=cfg/gym/finetune/ant-v2 \
        --config-name=ft_ppo_reflow_mlp \
        denoising_steps=${step} \
        ft_denoising_steps=${step} \
        device=cuda:0 \
        sim_device=cuda:0 \
        wandb=online

    echo "Finished experiment with denoising_steps=${step}"
    echo "================================================================"
done

echo "All ${#STEPS[@]} experiments completed successfully series!"
