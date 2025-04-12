import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class EdgeConv(MessagePassing):
    """Edge convolutional layer for circuit graphs."""
    def __init__(self, node_dim, edge_dim, out_dim):
        super(EdgeConv, self).__init__(aggr='add')
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.message_nn = nn.Sequential(
            nn.Linear(out_dim * 2 + edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.update_nn = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        node_features = self.node_encoder(x)
        
        # Start propagation
        return self.propagate(edge_index, x=x, node_features=node_features, edge_attr=edge_attr)
    
    def message(self, node_features_i, node_features_j, edge_attr):
        # Construct message from source node, destination node, and edge features
        edge_features = self.edge_encoder(edge_attr)
        msg_input = torch.cat([node_features_i, node_features_j, edge_attr], dim=1)
        return self.message_nn(msg_input)
    
    def update(self, aggr_out, x):
        # Update node features with aggregated messages
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_nn(update_input)


class TopologicalGNN(nn.Module):
    """Graph neural network that respects the DAG structure of circuits."""
    def __init__(self, input_node_dim, input_edge_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(TopologicalGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial projection
        self.node_encoder = nn.Linear(input_node_dim, hidden_dim)
        
        # Edge convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            self.convs.append(EdgeConv(in_dim, input_edge_dim, hidden_dim))
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Pooling layer for global representation
        self.pool = TopKPooling(hidden_dim, ratio=0.5)
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, input_node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, input_edge_dim]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            node_embeddings: Node-level embeddings [num_nodes, output_dim]
            global_embedding: Graph-level embedding [batch_size, output_dim]
        """
        # Initial projection
        x = self.node_encoder(x)
        
        # Store all intermediate representations
        node_embeddings = [x]
        
        # Message passing layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            node_embeddings.append(x)
        
        # Residual connection with initial features
        x = node_embeddings[0] + node_embeddings[-1]
        
        # Final node embeddings
        node_embeddings = self.output_layer(x)
        
        # Compute global representation if batch indices are provided
        if batch is not None:
            # Apply pooling
            pooled_x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
            
            # Global pooling operations
            mean_pool = global_mean_pool(pooled_x, batch)
            max_pool = global_max_pool(pooled_x, batch)
            sum_pool = global_add_pool(pooled_x, batch)
            
            # Combine different pooling results
            global_embedding = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
            global_embedding = self.global_pool(global_embedding)
            
            return node_embeddings, global_embedding
        
        return node_embeddings, None


class CircuitEncoder(nn.Module):
    """Circuit encoder model that processes circuit structures."""
    def __init__(self, input_node_dim=10, input_edge_dim=2, hidden_dim=128, output_dim=128, num_layers=3, dropout=0.1):
        super(CircuitEncoder, self).__init__()
        
        self.gnn = TopologicalGNN(
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass through the circuit encoder.
        
        Args:
            x: Node features [num_nodes, input_node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, input_edge_dim]
            batch: Batch assignment vector [num_nodes]
            
        Returns:
            node_embeddings: Node-level embeddings [num_nodes, output_dim]
            global_embedding: Graph-level embedding [batch_size, output_dim]
        """
        return self.gnn(x, edge_index, edge_attr, batch)
