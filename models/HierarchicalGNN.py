import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch

class HierarchicalGNN(nn.Module):
    def __init__(self,
                 node_dim: int = 32,  # Node feature dimension
                 hidden_dim: int = 128,  # Hidden layer dimension
                 num_heads: tuple = (3, 2, 2),  # Number of attention heads for site, route, and flight layers
                 dropout: float = 0.1):  # Dropout probability
        """
        Hierarchical Graph Neural Network module.
        
        Args:
            node_dim (int): Dimension of node features.
            hidden_dim (int): Dimension of hidden layers.
            num_heads (tuple): Tuple containing the number of heads for each GAT layer at different hierarchical levels.
            dropout (float): Dropout rate applied to layers.
        """
        super().__init__()
        
        # Site Graph Network Section
        self.site_gnn = nn.ModuleList([
            GATConv(in_channels=node_dim, 
                    out_channels=hidden_dim // num_heads[0],  # Ensuring consistent output dimension across heads
                    heads=num_heads[0], 
                    dropout=dropout),
            nn.BatchNorm1d(hidden_dim),  # Batch normalization
            nn.ELU(),  # Exponential Linear Unit activation
            GATConv(in_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    heads=1,  # Single head for the second layer
                    concat=False, 
                    dropout=dropout)  # No concatenation of heads' outputs
        ])
        
        # Route Graph Network Section
        self.route_gnn = nn.ModuleList([
            GATConv(node_dim, hidden_dim // num_heads[1], 
                    heads=num_heads[1], 
                    dropout=dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            GATConv(hidden_dim, hidden_dim, 
                    heads=1, 
                    concat=False, 
                    dropout=dropout)
        ])
        
        # Flight Graph Network Section
        self.flight_gnn = nn.ModuleList([
            GATConv(node_dim, hidden_dim // num_heads[2], 
                    heads=num_heads[2], 
                    dropout=dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            GATConv(hidden_dim, hidden_dim, 
                    heads=1, 
                    concat=False, 
                    dropout=dropout)
        ])
        
        # Similarity Constraint Section
        self.sim_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)  # Triplet loss for similarity constraint
        self.sim_proj = nn.Sequential(  # Projection for similarity computation in a lower-dimensional space
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 16)
        )
    
    def forward(self,
                site_data: Batch,  # Site graph data
                route_data: Batch,  # Route graph data
                flight_data: Batch):  # Flight graph data
        """
        Forward pass through the network.
        
        Args:
            site_data, route_data, flight_data: Batch objects containing graph data with x, edge_index, and batch attributes.
        
        Returns:
            tuple: A tuple containing combined features and similarity loss.

                - combined_feat: Aggregated features from all levels (batch_size, 3*hidden_dim).

                - sim_loss: Loss value from the similarity constraint.
        """
        
        # Site Feature Learning
        x = site_data.x
        edge_index = site_data.edge_index
        for layer in self.site_gnn:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        site_feat = x
        
        # Route Feature Learning (Similar process as site)
        x = route_data.x
        edge_index = route_data.edge_index
        for layer in self.route_gnn:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        route_feat = x
        
        # Flight Feature Learning (Following the same pattern)
        x = flight_data.x
        edge_index = flight_data.edge_index
        for layer in self.flight_gnn:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        flight_feat = x
        
        # Hierarchical Feature Aggregation
        site_pool = global_mean_pool(site_feat, site_data.batch)  # Mean pooling for site graphs
        route_pool = global_mean_pool(route_feat, route_data.batch)
        flight_pool = global_mean_pool(flight_feat, flight_data.batch)
        combined_feat = torch.cat([site_pool, route_pool, flight_pool], dim=1)  # Concatenation of features
        
        # Similarity Constraint Computation
        # Sampling triplets for similarity loss calculation
        anchor_idx = torch.randint(0, site_pool.size(0), (site_pool.size(0)//2,))
        pos_idx = (anchor_idx + 1) % site_pool.size(0)  # Positive samples
        neg_idx = torch.randint(0, site_pool.size(0), (anchor_idx.size(0),))  # Negative samples
        
        anchor = self.sim_proj(site_pool[anchor_idx])
        positive = self.sim_proj(site_pool[pos_idx])
        negative = self.sim_proj(site_pool[neg_idx])
        
        sim_loss = self.sim_loss_fn(anchor, positive, negative)
        
        return combined_feat, sim_loss
    
    def _init_weights(self):
        """
        Initialization of model weights.
        
        Applies Xavier initialization to GATConv layers and Kaiming initialization to Linear layers.
        """
        for m in self.modules():
            if isinstance(m, GATConv):
                nn.init.xavier_normal_(m.lin_src.weight)
                nn.init.xavier_normal_(m.lin_dst.weight)
                nn.init.xavier_normal_(m.att)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
