# ReinFlow 项目架构与启动全指南

本指南旨在帮助您快速配置环境、下载必要资产并启动 ReinFlow 训练。完成以下步骤后，您将能够复现论文中的实验结果。

---

## � 快速启动 (三步通关)

如果您刚刚 `git clone` 了本项目，可以直接使用以下简化流程：

### 1. 安装环境与所有依赖
现在的 `pip install` 会自动处理包括 `mjrl` 在内的所有复杂依赖。
```bash
conda create -n reinflow python=3.8 -y
conda activate reinflow
pip install -e .  # 这将一键安装 ReinFlow 及其所有外部 Git 依赖 (如 mjrl)
```

### 2. (可选) 自定义路径与资产
**路径优先级逻辑**：
*   **环境变量优先**：如果您在 `~/.bashrc` 中设置了 `REINFLOW_DATA_DIR`（例如指向 `/mnt/d/assets`），程序会优先读取。
*   **自动默认值**：如果没有设置环境变量，程序会自动在项目根目录下寻找 `data/` 和 `log/`。

**资产自动补齐**：
代码自带自动化下载功能。如果本地缺少必要的预训练权重或归一化数据，`script/run.py` 会尝试自动从云端拉取。

### 3. 直接启动训练
在激活环境后，您可以直接运行以下命令：
```bash
# 示例 1：启动 PPO 训练 (Ant 环境)
python script/run.py \
    --config-dir=cfg/gym/finetune/ant-v2 \
    --config-name=ft_ppo_reflow_mlp \
    wandb=null device=cuda:0 sim_device=cuda:0

# 示例 2：启动 FPO 训练 (Ant 环境)
python script/run.py \
    --config-dir=cfg/gym/finetune/ant-v2 \
    --config-name=ft_fpo_reflow_mlp \
    wandb=null device=cuda:0 sim_device=cuda:0
```

---

## 🛠️ 进阶：环境与路径管理 (传统方式)

如果您需要手动管理路径或资产，可以参考以下细节：

### 1. 基础引擎安装
*   **MuJoCo 210**: 需解压至 `~/.mujoco/mujoco210` 并配置 `LD_LIBRARY_PATH`。
*   **路径配置脚本**: 运行 `bash ./script/set_path.sh` 可手动指定 Data/Log 存放位置。

### 2. 下载必要资产 (Assets)
ReinFlow 的微调 (Fine-tuning) 流程依赖于 **预训练权重** 和 **归一化数据**，这些文件不包含在 Git 仓库中。
*   **归一化统计数据 (Normalization Data)**: 路径通常为 `${REINFLOW_DATA_DIR}/gym/${env_name}/normalization.npz`。
*   **预训练权重 (Pre-trained Checkpoints)**: 请确保您的 `.pt` 权重文件放在配置文件中 `base_policy_path` 指定的位置。

### 3. Weights & Biases 实验备注
本项目原生集成了自动给 wandb run 附加详细描述（Notes）的功能，以方便比较相似名称下不同配置的跑分结果：
* 您只需在项目根目录（即 `script/` 文件夹所在目录）新建或修改 `notes.txt`。
* 在执行任何带有 `wandb=online` 开启的训练脚本时，此文件的文字将被抓取并呈现在该运行大盘中的 Overview 面板下。

---

# 🏗️ 第四部分：项目架构说明

ReinFlow 采用模块化设计，结合 [Hydra](https://hydra.cc/) 实现算法与环境解耦。

## 1. 核心流程图

```mermaid
graph TD
    A[script/run.py] -- "1. 实例化" --> B[TrainPPOFlowAgent]
    B -- "2. 初始化" --> C[PPOFlow Model]
    B -- "2. 初始化" --> D[PPOFlowBuffer]
    
    subgraph "训练循环 (Iteration)"
        B -- "3. 采样" --> E[venv 并行环境]
        E -- "4. 返回经验" --> B
        B -- "5. 存储" --> D
        D -- "6. 整理批次" --> B
        B -- "7. 更新梯度" --> C
        C -- "8. 权重更新" --> F[Policy/Critic]
    end
```

## 2. 关键组件索引

| **训练循环 (PPO)** | `TrainPPOFlowAgent.run` | `agent/finetune/reinflow/train_ppo_flow_agent.py` |
| **训练循环 (FPO)** | `TrainFPOFlowAgent.run` | `agent/finetune/reinflow/train_fpo_flow_agent.py` |
| **PPO+FM 损失**| `PPOFlow.loss` | `model/flow/ft_ppo/ppoflow.py` |
| **FPO+FM 损失**| `FPOFlow.loss` | `model/flow/ft_ppo/fpoflow.py` |
| **动作生成流程** | `PPOFlow.get_actions` | `model/flow/ft_ppo/ppoflow.py` |
| **网络定义** | `FlowMLP` | `model/flow/mlp_flow.py` |
| **轨迹缓存** | `PPOFlowBuffer` | `agent/finetune/reinflow/buffer.py` |
