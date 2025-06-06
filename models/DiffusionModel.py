import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class DiffusionModel(nn.Module):
    def __init__(self,
                 cond_dim: int = 384,       # Dimension of conditional features (graph features + temporal)
                 seq_len: int = 30,         # Input sequence length
                 pred_steps: int = 7,       # Prediction steps
                 noise_dim: int = 64,       # Noise latent space dimension
                 num_timesteps: int = 1000, # Total diffusion steps
                 schedule: str = 'cosine'): # Noise scheduling strategy
        super().__init__()
        self.num_timesteps = num_timesteps
        self.pred_steps = pred_steps
        self.seq_len = seq_len
        
        # ================= Noise Scheduling Strategy =================
        self.register_buffer('betas', self._get_beta_schedule(schedule, num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
        
        # =============== Conditional Noise Prediction Network ================
        self.noise_pred_net = nn.Sequential(
            # Timestep embedding
            TimestepEmbedding(num_timesteps, 64),
            # Conditional feature processing
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            # Spatio-temporal cross-attention
            SpatialTemporalAttention(seq_len, noise_dim, 256),
            # Residual block
            ResidualBlock(256),
            # Output layer
            nn.Linear(256, seq_len)
        )
        
        # ================ Parameter Initialization ==================
        self._init_weights()

    def forward(self, 
                x: torch.Tensor,          # Input sequence [B, seq_len]
                cond_feat: torch.Tensor,   # Conditional features [B, cond_dim]
                noise: torch.Tensor = None,
                t: torch.Tensor = None) -> torch.Tensor:
        """
        Forward propagation during training
        Args:
            x: Original input sequence
            cond_feat: Conditional features (graph features + temporal features)
            noise: Optional pre-generated noise
            t: Optional specified timestep
        Returns:
            Predicted noise [B, seq_len]
        """
        # Randomly sample timestep
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.size(0), device=x.device))
            
        # Forward diffusion process
        x_noisy, noise_real = self.q_sample(x, t, noise)
        
        # Predict noise
        noise_pred = self.noise_pred_net(x_noisy, cond_feat, t)
        
        return noise_pred

    def q_sample(self,
                 x0: torch.Tensor,
                 t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion process (noise addition)
        Args:
            x0: Original sequence [B, seq_len]
            t: Timestep [B,]
            noise: Optional noise
        Returns:
            Noisy sequence [B, seq_len]
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_noisy = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def p_sample(self,
                 x: torch.Tensor,
                 cond_feat: torch.Tensor,
                 t: int) -> torch.Tensor:
        """
        Reverse denoising process (single step)
        Args:
            x: Current noisy sequence [B, seq_len]
            cond_feat: Conditional features [B, cond_dim]
            t: Current timestep
        Returns:
            Denoised sequence [B, seq_len]
        """
        # Predict noise
        noise_pred = self.noise_pred_net(x, cond_feat, t)
        
        # Calculate denoising result
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x_prev = (1 / sqrt(alpha_t)) * (x - beta_t / sqrt(1 - self.alphas_cumprod[t]) * noise_pred)
        x_prev += sqrt(beta_t) * noise
        
        return x_prev

    def predict(self,
                cond_feat: torch.Tensor) -> torch.Tensor:
        """
        Full reverse generation process
        Args:
            cond_feat: Conditional features [B, cond_dim]
        Returns:
            Predicted sequence [B, pred_steps]
        """
        # Initialize random noise
        x = torch.randn(
            (cond_feat.size(0), self.seq_len),
            device=cond_feat.device
        )
        
        # Stepwise denoising
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, cond_feat, t)
        
        # Extract prediction part
        return x[:, -self.pred_steps:]

    def _get_beta_schedule(self,
                           schedule: str,
                          num_timesteps: int) -> torch.Tensor:
        """ Generate beta scheduling table """
        if schedule == 'linear':
            return torch.linspace(1e-4, 0.02, num_timesteps)
        elif schedule == 'cosine':
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
            s = 0.008
            f = torch.cos((steps / num_timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
            return torch.clip(1 - f[1:] / f[:-1], 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _init_weights(self):
        """ Parameter initialization """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TimestepEmbedding(nn.Module):
    """ Timestep Embedding Module """
    def __init__(self, num_timesteps: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_timesteps, embed_dim)
        
    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.embed(t).unsqueeze(1)  # [B, 1, D]
        cond_feat = cond_feat.unsqueeze(1)    # [B, 1, D_cond]
        fused = torch.cat([x, t_embed, cond_feat], dim=-1)
        return fused

class SpatialTemporalAttention(nn.Module):
    """ Spatio-temporal Cross-Attention Module """
    def __init__(self, seq_len: int, noise_dim: int, feat_dim: int):
        super().__init__()
        self.query = nn.Linear(noise_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, D_noise]
        # cond_feat: [B, D_cond]
        B, L, D = x.shape
        
        # Expand conditional features
        cond_feat = cond_feat.unsqueeze(1).expand(-1, L, -1)  # [B, L, D_cond]
        
        # Compute attention
        q = self.query(x)  # [B, L, D]
        k = self.key(cond_feat)          
        v = self.value(cond_feat)
        
        attn = F.softmax(torch.bmm(q, k.transpose(1,2)) / sqrt(D)
        out = torch.bmm(attn, v) + x  # Residual connection
        return out

class ResidualBlock(nn.Module):
    """ Residual Block """
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
