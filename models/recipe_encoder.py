import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class RecipeGNN(MessagePassing):
    """GNN for encoding recipes."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(RecipeGNN, self).__init__(aggr='add')
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Message passing layers
        self.message_nns = nn.ModuleList()
        self.update_nns = nn.ModuleList()
        
        for i in range(num_layers):
            # Message function
            self.message_nns.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            
            # Update function
            self.update_nns.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the Recipe GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            x: Updated node features [num_nodes, output_dim]
        """
        # Initial projection
        x = self.input_proj(x)
        
        # Store initial features for residual connection
        x_init = x
        
        # Apply message passing layers
        for i in range(self.num_layers):
            # Message passing
            m = self.propagate(edge_index, x=x, message_nn=self.message_nns[i])
            
            # Update
            x = self.update_nns[i](torch.cat([x, m], dim=1))
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Residual connection
        x = x + x_init
        
        # Final projection
        x = self.output_proj(x)
        
        return x
    
    def message(self, x_i, x_j, message_nn):
        # Construct messages between nodes
        inputs = torch.cat([x_i, x_j], dim=1)
        return message_nn(inputs)


class RecipeEncoder(nn.Module):
    """Recipe encoder that processes recipe steps."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, num_layers=2, dropout=0.1):
        super(RecipeEncoder, self).__init__()
        
        self.recipe_gnn = RecipeGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the recipe encoder.
        
        Args:
            x: Recipe features [batch_size, max_steps, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment vector [batch_size * max_steps]
            
        Returns:
            x: Updated recipe features [batch_size, max_steps, output_dim]
        """
        batch_size, max_steps, feat_dim = x.size()
        
        # Reshape for GNN processing
        x_flat = x.view(-1, feat_dim)
        
        # Process through GNN
        x_flat = self.recipe_gnn(x_flat, edge_index, batch)
        
        # Reshape back
        x = x_flat.view(batch_size, max_steps, -1)
        
        return x
