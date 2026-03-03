import torch
from torch import nn
import torch.nn.functional as F
import logging
from model.flow.ft_ppo.ppoflow import PPOFlow

log = logging.getLogger(__name__)

class FPOFlow(PPOFlow):
    """
    Flow-matching Policy Optimization (FPO) Model.
    Inherits from PPOFlow and overrides the loss objective to match the FPO paper.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.info("Initialized FPOFlow model.")

    def compute_cfm_loss(self, cond, x1, eps, t):
        """
        Calculates the CFM loss on a given set of sampled noise and timesteps.
        cond: dict with 'state' of shape (B, cond_steps, obs_dim)
        x1: (B, horizon_steps, act_dim)
        eps: (B, N, horizon_steps, act_dim)
        t: (B, N, 1) -> will be used as time
        """
        B, N, H, D = eps.shape
        x1_exp = x1.unsqueeze(1).expand(B, N, H, D) # (B, N, H, D)
        
        # t is in [0, 1]
        t_exp = t.unsqueeze(-1) # (B, N, 1, 1)
        x_t = (1 - t_exp) * eps + t_exp * x1_exp
        
        # Flatten B and N
        x_t_flat = x_t.reshape(B*N, H, D)
        t_flat = t.reshape(B*N)
        
        # For cond, expand and flatten
        cond_flat = {}
        for k, v in cond.items():
            # v is (B, ...)
            cond_flat[k] = v.unsqueeze(1).expand(B, N, *v.shape[1:]).reshape(B*N, *v.shape[1:])
            
        # Target velocity
        target_v = x1_exp - eps
        target_v_flat = target_v.reshape(B*N, H, D)
        
        # Forward pass (do not learn exploration noise here since we just want the velocity)
        pred_v, _ = self.actor_ft.forward(action=x_t_flat, time=t_flat, cond=cond_flat, learn_exploration_noise=False)
        
        # CFM loss (MSE)
        cfm_loss_flat = F.mse_loss(pred_v, target_v_flat, reduction='none') # (B*N, H, D)
        cfm_loss = cfm_loss_flat.mean(dim=(1,2)).reshape(B, N) # (B, N)
        
        return cfm_loss

    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        loss_eps,
        loss_t,
        initial_cfm_loss,
        use_bc_loss=False,
        bc_loss_type='W2',
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True
    ):
        """
        FPO loss
        Here, B is the flattened batch size (usually minibatch_size).
        """
        # 1. Compute CFM loss for the current policy on the saved eps and t
        x1 = chains[:, -1] # (B, H, D)
        cfm_loss = self.compute_cfm_loss(obs, x1, loss_eps, loss_t) # (B, N)
        
        # 2. Policy ratio (rho_s)
        cfm_difference = initial_cfm_loss - cfm_loss
        cfm_difference = torch.clamp(cfm_difference, -3.0, 3.0)
        rho_s = torch.exp(torch.clamp(cfm_difference.mean(dim=1), -3.0, 3.0)) # (B,)
        
        # 3. Advantages normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean":f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0,1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")

        # 4. Policy loss
        pg_loss1 = -advantages * rho_s
        pg_loss2 = -advantages * torch.clamp(rho_s, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 5. Value loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef:
            v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")

        # Metrics
        with torch.no_grad():
            clipfrac = ((rho_s - 1.0).abs() > self.clip_ploss_coef).float().mean().item()
            
            # To avoid nan in log when rho_s is very small or zero, we clamp it slightly above 0.
            approx_kl = (rho_s - 1.0 - torch.log(torch.clamp(rho_s, min=1e-8))).mean().item()
            ratio_mean = rho_s.mean().item()
            
            # get dummy zero tensors for items expected by TrainPPOFlowAgent
            # like entropy loss, oldlogprobs
            entropy_loss = torch.tensor(0.0, device=self.device)
            oldlogprobs_min = torch.tensor(0.0)
            oldlogprobs_max = torch.tensor(0.0)
            oldlogprobs_std = torch.tensor(0.0)
            newlogprobs_min = torch.tensor(0.0)
            newlogprobs_max = torch.tensor(0.0)
            newlogprobs_std = torch.tensor(0.0)
            noise_std_item = 0.0

        return (
            pg_loss,
            entropy_loss, # entropy loss
            v_loss,
            0.0, # bc loss
            clipfrac,
            approx_kl,
            ratio_mean,
            oldlogprobs_min,
            oldlogprobs_max,
            oldlogprobs_std,
            newlogprobs_min,
            newlogprobs_max,
            newlogprobs_std,
            noise_std_item,
            newvalues.mean().item(),#Q function
        )
