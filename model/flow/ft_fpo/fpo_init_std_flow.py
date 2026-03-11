import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.normal import Normal
import logging
from typing import Tuple

from model.flow.ft_fpo.fpoflow import FPOFlow

log = logging.getLogger(__name__)

class FPOInitStdFlow(FPOFlow):
    """
    Flow-matching Policy Optimization (FPO) Model with adaptive initial std.
    Inherits from FPOFlow and adaptively scales the initial noise based on value estimates.
    """
    def __init__(self, std_min=0.4, std_max=1.5, std_lr=0.01, train_std_sync: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.std_min = std_min
        self.std_max = std_max
        self.std_lr = std_lr
        self.init_std = 1.0
        self.train_std_sync = train_std_sync
        log.info(f"Initialized FPOInitStdFlow with std_min={std_min}, std_max={std_max}, std_lr={std_lr}, train_std_sync={train_std_sync}")

    def compute_cfm_loss(self, cond, x1, eps, t):
        """
        Calculates the CFM loss on a given set of sampled noise and timesteps.
        """
        B, N, H, D = eps.shape
        x1_exp = x1.unsqueeze(1).expand(B, N, H, D)
        t_exp = t.unsqueeze(-1)
        
        # Apply init_std if train_std_sync is enabled, else use base N(0, 1) eps mapping
        if self.train_std_sync:
            x_0 = eps * self.init_std
        else:
            x_0 = eps
            
        x_t = (1 - t_exp) * x_0 + t_exp * x1_exp
        
        x_t_flat = x_t.reshape(B*N, H, D)
        t_flat = t.reshape(B*N)
        
        cond_flat = {}
        for k, v in cond.items():
            cond_flat[k] = v.unsqueeze(1).expand(B, N, *v.shape[1:]).reshape(B*N, *v.shape[1:])
            
        target_v = x1_exp - x_0
        target_v_flat = target_v.reshape(B*N, H, D)
        
        pred_v, _ = self.actor_ft.forward(action=x_t_flat, time=t_flat, cond=cond_flat, learn_exploration_noise=False)
        
        cfm_loss_flat = F.mse_loss(pred_v, target_v_flat, reduction='none')
        cfm_loss = cfm_loss_flat.mean(dim=(1,2)).reshape(B, N)
        return cfm_loss

    @torch.no_grad()
    def sample_first_point(self, B:int) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), self.init_std)
        xt = dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)
        xt = xt.reshape(B, self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob

    def get_logprobs(self, 
                     cond:dict, 
                     x_chain:Tensor, 
                     get_entropy=False, 
                     normalize_denoising_horizon=False, 
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        logprob = 0.0
        joint_entropy = 0.0 
        entropy_rate_est = 0.0
        logprob_steps = 0
        
        B = x_chain.shape[0]
        chains_prev = x_chain[:, :-1,:, :].flatten(-2,-1)
        chains_next = x_chain[:, 1:, :, :].flatten(-2,-1)
        chains_stds = torch.zeros_like(chains_prev, device=self.device)
        
        # logprob for initial state
        train_init_std = self.init_std if self.train_std_sync else 1.0
        init_dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), train_init_std)
        logprob_init = init_dist.log_prob(x_chain[:,0].reshape(B,-1)).sum(-1)
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)
        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1
        
        # transition probabilities
        chains_vel = torch.zeros_like(chains_prev, device=self.device)

        dt = 1.0/self.inference_steps
        steps = torch.linspace(0, 1-dt, self.inference_steps).repeat(B, 1).to(self.device)
        for i in range(self.inference_steps):
            t       = steps[:,i]
            xt      = x_chain[:,i]
            vt, nt  = self.actor_ft.forward(xt, t, cond, True, i)
            chains_vel[:,i]  = vt.flatten(-2,-1)
            chains_stds[:,i] = nt
            logprob_steps += 1
        chains_mean = (chains_prev + chains_vel * dt)
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        
        # transition distribution
        chains_dist = Normal(chains_mean, chains_stds)
        
        # logprobability and entropy of the transitions
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)
        
        logprob += logprob_trans.sum(-1)
        if self.logprob_debug_recalculate: 
            log.info(f"logprob_init={logprob_init.mean().item()}, logprob_trans={logprob_trans.mean().item()}")
        if get_entropy:
            joint_entropy += entropy_trans.sum(-1)
        
        if get_entropy:
            entropy_rate_est = joint_entropy/logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
            
        if normalize_act_space_dimension:
            logprob = logprob/self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est/self.act_dim_total
        
        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")
        
        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est, 
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob
