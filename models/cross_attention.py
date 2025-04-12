import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            q: Query tensor [batch_size, seq_len_q, d_model]
            k: Key tensor [batch_size, seq_len_k, d_model]
            v: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional mask [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attn: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        
        # Linear projections and split into heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Calculate output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out(output)
        
        return output, attn


class CrossAttention(nn.Module):
    """Cross-attention module for circuit-recipe interaction."""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, recipe_features, circuit_features, mask=None):
        """
        Forward pass through cross-attention.
        
        Args:
            recipe_features: Recipe features [batch_size, max_steps, d_model]
            circuit_features: Circuit node features [num_nodes, d_model]
            mask: Optional attention mask
            
        Returns:
            recipe_features: Updated recipe features [batch_size, max_steps, d_model]
            attn_weights: Attention weights
        """
        # Create attention mask if needed
        if mask is not None:
            # Expand mask for attention
            mask = mask.unsqueeze(-1).expand(-1, -1, circuit_features.size(0))
        
        # Cross-attention: recipe attends to circuit
        attn_output, attn_weights = self.attention(
            q=recipe_features,
            k=circuit_features.unsqueeze(0).expand(recipe_features.size(0), -1, -1),
            v=circuit_features.unsqueeze(0).expand(recipe_features.size(0), -1, -1),
            mask=mask
        )
        
        # Add & Norm
        recipe_features = self.norm1(recipe_features + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(recipe_features)
        
        # Add & Norm
        recipe_features = self.norm2(recipe_features + self.dropout(ff_output))
        
        return recipe_features, attn_weights
