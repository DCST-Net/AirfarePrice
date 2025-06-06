class HybridLoss(nn.Module):
    def __init__(self, 
                 alpha: float = 0.3,  # similarity loss weight
                 epsilon: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        
    def forward(self,
                pred: torch.Tensor,        # model predictions [B, T]
                target: torch.Tensor,      # true prices [B, T]
                sim_loss: torch.Tensor     # graph similarity loss
               ) -> torch.Tensor:
        """
        Hybrid loss calculation
        """
        # main loss function
        mae_loss = F.l1_loss(pred, target)
        
        # MAPE calculation (numerically stable version)
        abs_percent = torch.abs((target - pred) / (target + self.epsilon))
        mape_loss = torch.mean(abs_percent) * 100
        
        # total loss combination
        total_loss = mape_loss + 0.5*mae_loss + self.alpha*sim_loss
        
        return total_loss
