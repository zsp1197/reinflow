import os
import logging
import torch
import numpy as np
from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from agent.finetune.reinflow.buffer_fpo import FPOFlowBuffer

log = logging.getLogger(__name__)

class TrainFPOFlowAgent(TrainPPOFlowAgent):
    """
    TrainFPOFlowAgent: for training FPOFlow models.
    Inherits from TrainPPOFlowAgent to reuse the training loop,
    buffer management, and environment interaction logic.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        log.info("Initialized TrainFPOFlowAgent.")
        self.num_fpo_samples = cfg.train.get("num_fpo_samples", 50)

    def init_buffer(self):
        log.info("Initializing FPO Buffer")
        # specific to CPU buffer and non-image observations for ant-v2
        self.buffer = FPOFlowBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps=self.inference_steps,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
            num_fpo_samples=self.num_fpo_samples
        )

    def minibatch_generator(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs, loss_eps, loss_t, initial_cfm_loss = self.buffer.make_dataset()
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        self.total_steps = self.n_steps * self.n_envs
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                minibatch = (
                    {"state": obs[inds_b]},
                    chains[inds_b],
                    returns[inds_b], 
                    oldvalues[inds_b],
                    advantages[inds_b],
                    oldlogprobs[inds_b],
                    loss_eps[inds_b],
                    loss_t[inds_b],
                    initial_cfm_loss[inds_b]
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl:
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch
                
    def minibatch_generator_repeat(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs, loss_eps, loss_t, initial_cfm_loss = self.buffer.make_dataset()
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        duplicate_multiplier = 10
        self.total_steps = self.n_steps * self.n_envs * duplicate_multiplier
        
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, duplicate_multiplier),
                )
                minibatch = (
                    {"state": obs[batch_inds_b]},
                    chains[batch_inds_b],
                    returns[batch_inds_b], 
                    oldvalues[batch_inds_b],
                    advantages[batch_inds_b],
                    oldlogprobs[batch_inds_b],
                    loss_eps[batch_inds_b],
                    loss_t[batch_inds_b],
                    initial_cfm_loss[batch_inds_b] 
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl:
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch

    def agent_update(self, verbose=True):
        # Generate the FPO required noise and loss traits right before learning
        log.info("Computing initial CFM loss over the collected trajectories...")
        self.buffer.update_fpo_trajs(self.model)
        
        super().agent_update(verbose)
