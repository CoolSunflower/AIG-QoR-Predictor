import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.circuit_encoder import CircuitEncoder
from models.recipe_encoder import RecipeEncoder
from models.cross_attention import CrossAttention
from models.prediction_models import TransformerPredictor, LSTMPredictor


class QoRPredictor(nn.Module):
    """Full QoR prediction model."""
    def __init__(
        self,
        node_feature_dim=10,
        edge_feature_dim=2,
        recipe_feature_dim=9,  # 8 for one-hot + 1 for position
        hidden_dim=128,
        embedding_dim=128,
        circuit_gnn_layers=3,
        recipe_gnn_layers=2,
        transformer_layers=4,
        lstm_layers=2,
        attention_heads=8,
        dropout=0.1
    ):
        super(QoRPredictor, self).__init__()
        
        # Circuit encoder
        self.circuit_encoder = CircuitEncoder(
            input_node_dim=node_feature_dim,
            input_edge_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=circuit_gnn_layers,
            dropout=dropout
        )
        
        # Recipe encoder
        self.recipe_encoder = RecipeEncoder(
            input_dim=recipe_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=recipe_gnn_layers,
            dropout=dropout
        )
        
        # Cross-attention for circuit-recipe interaction
        self.cross_attention = CrossAttention(
            d_model=embedding_dim,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Circuit global context projection
        self.global_context_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer predictor
        self.transformer_predictor = TransformerPredictor(
            d_model=embedding_dim,
            nhead=attention_heads,
            num_layers=transformer_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        
        # LSTM predictor
        self.lstm_predictor = LSTMPredictor(
            d_model=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # Output combiner
        self.output_combiner = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, p in self.named_parameters():
            if 'lstm' in name and 'weight' in name:
                nn.init.orthogonal_(p)
            elif len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
    
    def encode_circuits(self, circuit_batch):
        """
        Encode circuit graphs.
        
        Args:
            circuit_batch: Batched PyTorch Geometric data object
            
        Returns:
            node_embeddings: Node-level embeddings
            global_embeddings: Graph-level embeddings
        """
        return self.circuit_encoder(
            circuit_batch.x,
            circuit_batch.edge_index,
            circuit_batch.edge_attr,
            circuit_batch.batch
        )
    
    def encode_recipes(self, recipe_features, recipe_edge_index):
        """
        Encode recipes.
        
        Args:
            recipe_features: Recipe features [batch_size, max_steps, feature_dim]
            recipe_edge_index: Edge indices for recipe graph
            
        Returns:
            recipe_embeddings: Recipe embeddings [batch_size, max_steps, embedding_dim]
        """
        return self.recipe_encoder(recipe_features, recipe_edge_index)
    
    def forward(self, batch):
        """
        Forward pass through the full QoR predictor model.
        
        Args:
            batch: Dictionary containing:
                - circuit_data: PyG batch of circuit graphs
                - recipe_features: Recipe features [batch_size, max_steps, feature_dim]
                - recipe_edge_index: Edge indices for recipe graph
                - design_indices: Indices mapping batches to unique designs [batch_size]
                - step_masks: Masks for valid recipe steps [batch_size, max_steps]
                
        Returns:
            predictions: Dictionary containing:
                - transformer_pred: Transformer predictions [batch_size, max_steps]
                - lstm_pred: LSTM predictions [batch_size, max_steps]
                - combined_pred: Combined predictions [batch_size, max_steps]
                - attention_weights: Attention weights from cross-attention
        """
        # Extract batch components
        circuit_data = batch['circuit_data']
        recipe_features = batch['recipe_features']
        recipe_edge_index = batch['recipe_edge_index']
        design_indices = batch['design_indices']
        step_masks = batch['step_masks']
        
        # Encode circuits - this gives embeddings for unique designs
        node_embeddings, global_embeddings = self.encode_circuits(circuit_data)
        
        # Encode recipes
        recipe_embeddings = self.encode_recipes(recipe_features, recipe_edge_index)
        
        # Get global circuit context for each batch item
        batch_size = recipe_features.size(0)
        circuit_context = global_embeddings[design_indices]
        circuit_context = self.global_context_proj(circuit_context)
        
        # For each batch item, get the corresponding circuit node embeddings
        # This requires mapping from batch indices to the actual node embeddings
        batch_node_embeddings = []
        ptr = circuit_data.ptr.cpu().numpy()  # Get pointers to batch slices
        
        for i, design_idx in enumerate(design_indices):
            # Get start and end indices for this design
            start_idx = ptr[design_idx]
            end_idx = ptr[design_idx + 1]
            
            # Get node embeddings for this design
            design_node_emb = node_embeddings[start_idx:end_idx]
            batch_node_embeddings.append(design_node_emb)
        
        # Cross-attention between recipes and circuit nodes
        enhanced_recipes = []
        attention_weights = []
        
        for i in range(batch_size):
            # Apply cross-attention between recipe features and circuit nodes
            enhanced_recipe, attn_weight = self.cross_attention(
                recipe_embeddings[i].unsqueeze(0),
                batch_node_embeddings[i],
                mask=step_masks[i].unsqueeze(0) if step_masks is not None else None
            )
            enhanced_recipes.append(enhanced_recipe)
            attention_weights.append(attn_weight)
        
        # Stack enhanced recipe embeddings
        enhanced_recipes = torch.cat(enhanced_recipes, dim=0)
        
        # Predictions from transformer
        transformer_pred = self.transformer_predictor(
            enhanced_recipes,
            mask=step_masks if step_masks is not None else None
        )
        
        # Predictions from LSTM
        lstm_pred = self.lstm_predictor(
            enhanced_recipes,
            circuit_context=circuit_context,
            mask=step_masks if step_masks is not None else None
        )
        
        # Combine predictions
        if step_masks is not None:
            # Ensure predictions respect the mask
            transformer_pred = transformer_pred.masked_fill(~step_masks, 0)
            lstm_pred = lstm_pred.masked_fill(~step_masks, 0)
            
            # Stack predictions for combining
            stacked_preds = torch.stack([transformer_pred, lstm_pred], dim=-1)
            
            # Apply combiner where mask is True, otherwise keep zeros
            combined_mask = step_masks.unsqueeze(-1).expand_as(stacked_preds)
            combined_pred = torch.zeros_like(transformer_pred)
            
            # Only apply combiner where mask is True
            valid_preds = stacked_preds[combined_mask.bool()].view(-1, 2)
            if valid_preds.size(0) > 0:  # Check if there are any valid predictions
                combined_valid = self.output_combiner(valid_preds).view(-1)
                combined_pred[step_masks] = combined_valid
        else:
            # Stack predictions for combining
            stacked_preds = torch.stack([transformer_pred, lstm_pred], dim=-1)
            combined_pred = self.output_combiner(stacked_preds).squeeze(-1)
        
        return {
            'transformer_pred': transformer_pred,
            'lstm_pred': lstm_pred,
            'combined_pred': combined_pred,
            'attention_weights': attention_weights
        }
