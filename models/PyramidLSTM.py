import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidLSTM(nn.Module):
    def __init__(self,
                 input_dim: int = 385,         # Input dimension (graph features 384 + time series feature 1)
                 pred_steps: int = 7,          # Prediction steps
                 pyramid_levels: list = [7, 14, 21],  # Multi-scale lookback periods
                 lstm_hidden: int = 64,        # LSTM hidden units
                 num_layers: int = 2,          # Number of LSTM layers
                 use_attention: bool = True):   # Whether to use attention fusion
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.use_attention = use_attention
        
        # ================= Multi-scale LSTM Network =================
        self.lstm_towers = nn.ModuleList([
            nn.LSTM(input_size=input_dim,
                    hidden_size=lstm_hidden,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=False)
            for _ in pyramid_levels
        ])
        
        # ================= Feature Fusion Module ==================
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden,
                num_heads=4,
                dropout=0.1
            )
            self.fusion = nn.Sequential(
                nn.Linear(lstm_hidden * len(pyramid_levels), 128),
                nn.ReLU(),
                nn.Linear(128, pred_steps)
            )
        else:
            self.fusion = nn.Linear(lstm_hidden * len(pyramid_levels), pred_steps)
            
        # ================ Temporal Positional Encoding ================
        self.pos_encoder = PositionalEncoding(lstm_hidden)

    def forward(self,
                 x: torch.Tensor,              # Input time series data [B, L]
                 graph_feat: torch.Tensor      # Graph features [B, 384]
               ) -> torch.Tensor:
        """
        Forward propagation
        Args:
            x: Time series data [batch_size, seq_len]
            graph_feat: Graph features [batch_size, 384]
        Returns:
            Periodic prediction result [batch_size, pred_steps]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Feature concatenation (time series data + graph features)
        x = x.unsqueeze(-1)  # [B, L, 1]
        graph_feat = graph_feat.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, 384]
        combined = torch.cat([x, graph_feat], dim=-1)  # [B, L, 385]
        
        # Multi-scale feature extraction
        pyramid_features = []
        for level, lstm in zip(self.pyramid_levels, self.lstm_towers):
            # Downsampling processing
            pooled = self._temporal_pooling(combined, level)  # [B, L//level, 385]
            
            # Positional encoding enhancement
            pooled = self.pos_encoder(pooled)
            
            # LSTM processing
            lstm_out, _ = lstm(pooled)  # [B, L//level, lstm_hidden]
            
            # Take last timestep feature
            last_step = lstm_out[:, -1, :]  # [B, lstm_hidden]
            pyramid_features.append(last_step)
        
        # Multi-scale feature fusion
        if self.use_attention:
            # Attention fusion
            attn_in = torch.stack(pyramid_features, dim=1)  # [B, num_levels, lstm_hidden]
            attn_out, _ = self.attention(
                attn_in.transpose(0,1),  # [num_levels, B, lstm_hidden]
                attn_in.transpose(0,1),
                attn_in.transpose(0,1)
            )  # [num_levels, B, lstm_hidden]
            fused = attn_out.transpose(0,1).reshape(batch_size, -1)  # [B, num_levels*lstm_hidden]
        else:
            # Direct concatenation
            fused = torch.cat(pyramid_features, dim=-1)  # [B, num_levels*lstm_hidden]
        
        # Final prediction
        return self.fusion(fused)  # [B, pred_steps]

    def _temporal_pooling(self,
                          x: torch.Tensor,
                          pool_size: int) -> torch.Tensor:
        """
        Temporal pyramid pooling
        Args:
            x: Input features [B, L, D]
            pool_size: Pooling window size
        Returns:
            Pooled features [B, L//pool_size, D]
        """
        L = x.size(1)
        if L % pool_size != 0:
            pad_size = pool_size - (L % pool_size)
            x = F.pad(x, (0,0,0,pad_size))
        
        # Reshape to 3D tensor for pooling [B, L/pool_size, pool_size, D]
        x_pool = x.view(x.size(0), -1, pool_size, x.size(2))
        # Average pooling [B, new_L, D]
        return x_pool.mean(dim=2)

class PositionalEncoding(nn.Module):
    """ Learnable positional encoding """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.position_emb = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        pos_emb = self.position_emb(positions).unsqueeze(0)
        return x + pos_emb
