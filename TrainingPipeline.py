import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TrainingPipeline:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 init_lr: float = 1e-4,
                 early_stop: int = 10):
        """
        Complete training management class
        
        Args:
            model: Model to be trained
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
            init_lr: Initial learning rate
            early_stop: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.early_stop = early_stop
        
        # Optimizer configuration
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=init_lr,
            weight_decay=1e-5
        )
        
        # Learning rate scheduling
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss function
        self.criterion = HybridLoss(alpha=0.3)
        
        # Training state tracking
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

    def train_epoch(self) -> float:
        """ Single training epoch """
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            # Unpack data (adjust according to actual data format)
            (site_data, route_data, flight_data), ts_a, ts_b, targets = batch
            ts_a = ts_a.to(self.device)
            ts_b = ts_b.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            graph_feat, sim_loss = self.model.gnn(site_data, route_data, flight_data)
            trend_pred, periodic_pred = self.model.decoder(ts_a, ts_b, graph_feat)
            final_pred = self.model.fusion(trend_pred, periodic_pred)
            
            # Loss calculation
            loss = self.criterion(final_pred, targets, sim_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """ Validation step """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                (site_data, route_data, flight_data), ts_a, ts_b, targets = batch
                ts_a = ts_a.to(self.device)
                ts_b = ts_b.to(self.device)
                targets = targets.to(self.device)
                
                graph_feat, sim_loss = self.model.gnn(site_data, route_data, flight_data)
                trend_pred, periodic_pred = self.model.decoder(ts_a, ts_b, graph_feat)
                final_pred = self.model.fusion(trend_pred, periodic_pred)
                
                loss = self.criterion(final_pred, targets, sim_loss)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)

    def run(self, epochs: int = 100):
        """ Complete training loop """
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Learning rate adjustment
            self.scheduler.step(val_loss)
            
            # Early stopping mechanism
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                torch.save(self.model.state_dict(), f'best_model_epoch{epoch}.pt')
            else:
                self.no_improve_epochs += 1
                if self.no_improve_epochs >= self.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Training status output
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("-"*50)
