import torch 
from torch import nn as nn
import numpy as np

class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
    def add_noise(self, x_0, t=None, noise=None):
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.uniform_sample_t(batch_size)
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        if noise is None:
            noise = torch.randn_like(x_0)  # (B, N, d)
            
        return c0 * x_0 + c1 * noise, beta
    
    def sample(self, context, flexibility=0.0, ret_traj=False, intermediate_steps=0, net=None, noise=None):
        batch_size = context.size(0)

        x_T = noise

        traj = {self.num_steps: x_T}
        intermediates = []

        for t in range(self.num_steps, 0, -1):            
            x_next = self.step(traj[t], t, context, net, flexibility)
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
            if intermediate_steps > 0 and t % (self.num_steps//intermediate_steps) == 0:
                intermediates.append(x_next)
        
        if ret_traj:
            if intermediate_steps > 0:
                return traj, intermediates
            return traj
        else:
            if intermediate_steps > 0:
                return traj[0], intermediates
            return traj[0]

        # for t in range(self.num_steps, 0, -1):
        #     z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        #     alpha = self.alphas[t]
        #     alpha_bar = self.alpha_bars[t]
        #     sigma = self.get_sigmas(t, flexibility)

        #     c0 = 1.0 / torch.sqrt(alpha)
        #     c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

        #     x_t = traj[t]
        #     beta = self.betas[[t]*batch_size]
        #     e_theta = net(x_t, beta=beta, context=context)
        #     x_next = c0 * (x_t - c1 * e_theta) + sigma * z
        #     traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
        #     traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
        #     if not ret_traj:
        #         del traj[t]
        #     if intermediate_steps > 0 and t % (self.num_steps//intermediate_steps) == 0:
        #         intermediates.append(x_next)
        
        # if ret_traj:
        #     if intermediate_steps > 0:
        #         return traj, intermediates
        #     return traj
        # else:
        #     if intermediate_steps > 0:
        #         return traj[0], intermediates
        #     return traj[0]
    
    def step(self, x_t, t, context, net, flexibility=0.0):

        batch_size = context.size(0)
        z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        sigma = self.get_sigmas(t, flexibility)

        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        
        beta = self.betas[[t]*batch_size]
        e_theta = net(x_t, beta=beta, context=context)
        return c0 * (x_t - c1 * e_theta) + sigma * z
        


class SaxiNoiseScheduler(nn.Module):
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        """
        Args:
            num_timesteps (int): Number of timesteps in the diffusion process.
            beta_start (float): Starting value of beta (variance).
            beta_end (float): Ending value of beta (variance).
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linearly interpolate beta values
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        
        # Compute alpha and alpha_bar
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, axis=0)  # Cumulative product

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

    def get_alpha_bar(self, t: int):
        """Get alpha_bar for a specific timestep."""
        return self.alpha_bars[t]

    def add_noise(self, x, noise, t):
        """
        Add noise to input x at timestep t.
        
        Args:
            x (torch.Tensor): Original image tensor.                
            noise (torch.Tensor): Noise to add.
            t (int): Timestep index.
        
        Returns:
            torch.Tensor: Noisy image tensor.
        """
        alpha_bar_t = self.get_alpha_bar(t)[:, None, None]

        while len(alpha_bar_t.shape) < len(x.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

    def step(self, x, noise_pred, timestep):
        """
        Perform one step of denoising on the noisy input x.
        
        Args:
            x (torch.Tensor): Noisy input at timestep t.
            noise_pred (torch.Tensor): Predicted noise (epsilon_pred) from the model.
            timestep (int): Current timestep.
        
        Returns:
            torch.Tensor: Updated input x after one step of denoising.
        """            
        alpha_t = self.alphas[timestep]
        beta_t = self.betas[timestep]
        alpha_bar_t = self.alpha_bars[timestep]

        # Ensure alpha_t and beta_t dimensions match x
        while len(alpha_t.shape) < len(x.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        while len(beta_t.shape) < len(x.shape):
            beta_t = beta_t.unsqueeze(-1)

        # Reconstruct x_start (denoised image estimate)
        x_start = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        # Compute x_{t-1}
        if timestep > 0:
            noise = torch.randn_like(x)  # Sample random noise
            x_next = torch.sqrt(alpha_t) * x_start + torch.sqrt(beta_t) * noise
        else:
            x_next = x_start  # No noise added at t=0

        return x_next
        

