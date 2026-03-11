import os
import logging
import torch
import numpy as np
from agent.finetune.reinflow.train_fpo_flow_agent import TrainFPOFlowAgent

log = logging.getLogger(__name__)

class TrainFPOInitStdFlowAgent(TrainFPOFlowAgent):
    """
    TrainFPOInitStdFlowAgent: for training FPOInitStdFlow models.
    Inherits from TrainFPOFlowAgent and dynamically adjusts `init_std` based on value network outputs.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        log.info("Initialized TrainFPOInitStdFlowAgent.")

    def agent_update(self, verbose=True):
        super().agent_update(verbose)
        
        # Access the average Q-value of the current minibatch evaluated in the PPO loop
        if hasattr(self, "train_ret_dict") and "Q_values" in self.train_ret_dict:
            mean_value = self.train_ret_dict["Q_values"]
            
            # Extract current model param bounds
            std_min = self.model.std_min
            std_max = self.model.std_max
            std_lr = self.model.std_lr
            
            # Emulate SAC-like adaptive mechanism: 
            # if mean_value is close to 1, target is std_max (explore more).
            # if mean_value is close to 0 or negative, target is std_min (exploit more).
            weight = np.clip(mean_value, 0.0, 1.0)
            target_std = std_min + (std_max - std_min) * float(weight)
            
            # Smooth EMA update of init_std
            self.model.init_std = (1.0 - std_lr) * self.model.init_std + std_lr * target_std
            
            if verbose:
                log.info(f"Updated init_std to {self.model.init_std:.4f} (target: {target_std:.4f}, mean_value: {mean_value:.4f})")
            
            # Log the adaptive init_std dynamically adjusted by the model
            self.train_ret_dict["fpo_init_std"] = self.model.init_std

