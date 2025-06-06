import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossFusion(nn.Module):
    def __init__(self,
                 trend_dim: int = 7,      # Trend prediction dimension
                 period_dim: int = 7,     # Period prediction dimension
                 hidden_dim: int = 32,    # Hidden dimension
                 num_heads: int = 2,      # Number of attention heads
                 use_residual: bool = True # Whether to use residual connection
                ):
        super().__init__()
        self.use_residual = use_residual
        
        # =============== Cross-attention mechanism ===============
        self.trend_proj = nn.Linear(trend_dim, hidden_dim)
        self.period_proj = nn.Linear(period_dim, hidden_dim) 
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        ) 
        
        # =============== Gated fusion module ================
        self.gate_net = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Sigmoid()
        ) 
        
        # =============== Output projection layer ================
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trend_dim)  # Keep output dimension consistent with input
        ) 
        
        # =============== Normalization layer ================
        self.layer_norm = nn.LayerNorm(hidden_dim) 
        
        # Initialize parameters
        self._init_weights()

    def forward(self,
               trend_feat: torch.Tensor,  # Trend branch prediction [B, T]
               period_feat: torch.Tensor  # Period branch prediction [B, T]
              ) -> torch.Tensor:
        """
        Dual-channel feature fusion
        Args:
            trend_feat: Trend branch prediction result [batch_size, pred_steps]
            period_feat: Period branch prediction result [batch_size, pred_steps]
        Returns:
            Fused prediction result [batch_size, pred_steps]
        """
        # Feature projection
        Q = self.trend_proj(trend_feat.unsqueeze(1))  # [B, 1, D]
        K = self.period_proj(period_feat.unsqueeze(1))
        V = K 
        
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=Q,
            key=K,
            value=V
        )  # [B, 1, D] 
        
        # Gated fusion
        trend_flat = self.trend_proj(trend_feat)  # [B, D]
        period_flat = self.period_proj(period_feat)
        combined = torch.cat([trend_flat, period_flat], dim=-1) 
        
        gate = self.gate_net(combined)  # [B, D]
        fused = gate * trend_flat + (1 - gate) * period_flat 
        
        # Residual connection
        if self.use_residual:
            fused = fused + attn_out.squeeze(1) 
        
        # Output projection
        fused = self.layer_norm(fused)
        output = self.output_proj(fused) 
        
        return output

    def _init_weights(self):
        """ Parameter initialization """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def get_attention_map(self,
                         trend_feat: torch.Tensor,
                         period_feat: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights (for visualization)
        Returns:
            Attention weight matrix [B, num_heads, 1, 1]
        """
        Q = self.trend_proj(trend_feat.unsqueeze(1))
        K = self.period_proj(period_feat.unsqueeze(1)) 
        
        _, attn_weights = self.cross_attn(
            Q, K, K,
            average_attn_weights=False
        )
        return attn_weights
