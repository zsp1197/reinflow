#!/bin/bash

# 本脚本用于研究 ReinFlow 在固定推理精度（inference_steps=9）时，不同的微调更新步数（ft_denoising_steps）对训练效果的影响。
# 依次测试的 ft_denoising_steps 值为: 1, 3, 5, 7, 9
# 推理步数固定为: 9

FT_STEPS=(1 3 5 7 9)
FIXED_INF_STEPS=9

echo "Starting sequential FT Denoising Steps comparison experiments..."

for ft_step in "${FT_STEPS[@]}"; do
    echo "================================================================"
    echo "Starting experiment with ft_denoising_steps=${ft_step} (Inference=${FIXED_INF_STEPS})"
    echo "================================================================"

    # 动态写入 notes.txt，增加关于步数解耦的备注
    cat <<EOF > notes.txt
[Decoupling Experiment]
Environment: ant-medium-expert-v2 (Gym)
Policy: ft_ppo_reflow_mlp (PPOFlow)
Total Inference Steps (Sampling): ${FIXED_INF_STEPS}
FT Denoising Steps (Updates): ${ft_step}

Objective: This experiment evaluates the impact of training only the last ${ft_step} steps of a ${FIXED_INF_STEPS}-step denoising trajectory.
EOF

    # 开始执行当前参数下的训练配置
    # 注意：PYTHONPATH=$(pwd) 确保脚本能找到 util 模块
    python script/run.py \
        --config-dir=cfg/gym/finetune/ant-v2 \
        --config-name=ft_ppo_reflow_mlp \
        run_name="ft${ft_step}_inf${FIXED_INF_STEPS}" \
        denoising_steps=${FIXED_INF_STEPS} \
        ft_denoising_steps=${ft_step} \
        device=cuda:0 \
        sim_device=cuda:0 \
        wandb=online

    echo "Finished experiment with ft_denoising_steps=${ft_step}"
    echo "================================================================"
done

echo "All ${#FT_STEPS[@]} experiments in the FT-decoupling series completed successfully!"
