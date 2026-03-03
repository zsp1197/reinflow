import torch
import numpy as np
import logging
from agent.finetune.reinflow.buffer import PPOFlowBuffer, PPOFlowBufferGPU

log = logging.getLogger(__name__)

class FPOFlowBuffer(PPOFlowBuffer):
    def __init__(self,
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device,
                 num_fpo_samples=50):
        super().__init__(
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device)
                 
        self.num_fpo_samples = num_fpo_samples

    def reset(self):
        super().reset()
        self.loss_eps_trajs = np.zeros((self.n_steps, self.n_envs, self.num_fpo_samples, self.horizon_steps, self.action_dim), dtype=np.float32)
        self.loss_t_trajs = np.zeros((self.n_steps, self.n_envs, self.num_fpo_samples, 1), dtype=np.float32)
        self.initial_cfm_loss_trajs = np.zeros((self.n_steps, self.n_envs, self.num_fpo_samples), dtype=np.float32)

    def make_dataset(self):
        obs, chains, returns, values, advantages, logprobs = super().make_dataset()
        
        loss_eps = torch.tensor(self.loss_eps_trajs, device=self.device).float().flatten(0, 1)
        loss_t = torch.tensor(self.loss_t_trajs, device=self.device).float().flatten(0, 1)
        initial_cfm_loss = torch.tensor(self.initial_cfm_loss_trajs, device=self.device).float().flatten(0, 1)

        return obs, chains, returns, values, advantages, logprobs, loss_eps, loss_t, initial_cfm_loss

    @torch.no_grad()
    def update_fpo_trajs(self, model):
        # Move required trajectories to GPU locally for computation
        for step in range(self.n_steps):
            cond = {
                "state": torch.from_numpy(self.obs_trajs["state"][step]).float().to(self.device)
            }

            
            critic = model.critic
            self.value_trajs[step] = critic.forward(cond).cpu().numpy().flatten()
            
            chains_venv = torch.from_numpy(self.chains_trajs[step]).float().to(self.device)
            x1 = chains_venv[:, -1] 
            
            # FPO specific sampling
            eps = torch.randn(self.n_envs, self.num_fpo_samples, self.horizon_steps, self.action_dim, device=self.device)
            t = torch.rand(self.n_envs, self.num_fpo_samples, 1, device=self.device)
            
            self.loss_eps_trajs[step] = eps.cpu().numpy()
            self.loss_t_trajs[step] = t.cpu().numpy()
            self.initial_cfm_loss_trajs[step] = model.compute_cfm_loss(cond, x1, eps, t).cpu().numpy()
            
            # FPO doesn't strictly need logprobs
            self.logprobs_trajs[step] = 0.0

class FPOFlowBufferGPU(PPOFlowBufferGPU):
    def __init__(self,
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device,
                 num_fpo_samples=50):
        super().__init__(
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device)
                 
        self.num_fpo_samples = num_fpo_samples

    def reset(self):
        super().reset()
        self.loss_eps_trajs = torch.zeros((self.n_steps, self.n_envs, self.num_fpo_samples, self.horizon_steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.loss_t_trajs = torch.zeros((self.n_steps, self.n_envs, self.num_fpo_samples, 1), dtype=torch.float32, device=self.device)
        self.initial_cfm_loss_trajs = torch.zeros((self.n_steps, self.n_envs, self.num_fpo_samples), dtype=torch.float32, device=self.device)

    def make_dataset(self):
        # We need to return the FPO specific items as well
        obs = self.obs_trajs["state"].flatten(0, 1)
        chains = self.chains_trajs.flatten(0, 1)
        returns = self.returns_trajs.flatten(0, 1)
        values = self.value_trajs.flatten(0, 1)
        advantages = self.advantages_trajs.flatten(0, 1)
        logprobs = self.logprobs_trajs.flatten(0, 1)
        
        loss_eps = self.loss_eps_trajs.flatten(0, 1)
        loss_t = self.loss_t_trajs.flatten(0, 1)
        initial_cfm_loss = self.initial_cfm_loss_trajs.flatten(0, 1)

        return obs, chains, returns, values, advantages, logprobs, loss_eps, loss_t, initial_cfm_loss

    @torch.no_grad()
    def update_value_logprob(self, model):
        """
        In FPO, we override this to compute value and initial_cfm_loss.
        We also generate noise and t for the FPO loss.
        """
        for step in range(self.n_steps):
            cond = {
                key: self.obs_trajs[key][step].to(self.device) for key in self.obs_dim
            }
            
            critic = model.critic
            self.value_trajs[step] = critic.forward(cond, no_augment=True).squeeze().float().to(self.device)
            
            chains_venv = self.chains_trajs[step] # (n_envs, ft_denoising_steps+1, horizon_steps, action_dim)
            x1 = chains_venv[:, -1] # The final generated action
            
            # FPO specific sampling
            eps = torch.randn(self.n_envs, self.num_fpo_samples, self.horizon_steps, self.action_dim, device=self.device)
            t = torch.rand(self.n_envs, self.num_fpo_samples, 1, device=self.device)
            
            self.loss_eps_trajs[step] = eps
            self.loss_t_trajs[step] = t
            self.initial_cfm_loss_trajs[step] = model.compute_cfm_loss(cond, x1, eps, t).detach()
            
            # We don't strictly need logprobs for FPO update, zero them out to save time
            self.logprobs_trajs[step] = 0.0
