import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPredictor(nn.Module):
    """Transformer-based predictor for step-wise AND gate counts."""
    def __init__(self, d_model, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Output projections
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass through transformer predictor.
        
        Args:
            x: Recipe features [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            outputs: Predicted values [batch_size, seq_len, 1]
        """
        # Create attention mask for transformer
        if mask is not None:
            # Convert True/False mask to 0.0/-inf mask for transformer
            attn_mask = ~mask.unsqueeze(1).expand(-1, mask.size(1), -1)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
        else:
            attn_mask = None
        
        # Apply transformer encoder
        outputs = self.transformer_encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        
        # Apply output projection
        outputs = self.output_layer(outputs)
        
        return outputs.squeeze(-1)


class LSTMPredictor(nn.Module):
    """LSTM-based predictor for step-wise AND gate counts."""
    def __init__(self, d_model, hidden_dim=256, num_layers=2, dropout=0.1):
        super(LSTMPredictor, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize hidden state projections
        self.init_h = nn.Linear(d_model, hidden_dim * num_layers * 2)
        self.init_c = nn.Linear(d_model, hidden_dim * num_layers * 2)
    
    def forward(self, x, circuit_context=None, mask=None):
        """
        Forward pass through LSTM predictor.
        
        Args:
            x: Recipe features [batch_size, seq_len, d_model]
            circuit_context: Optional circuit context vector [batch_size, d_model]
            mask: Sequence mask [batch_size, seq_len]
            
        Returns:
            outputs: Predicted values [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states with circuit context if provided
        if circuit_context is not None:
            h0 = self.init_h(circuit_context).view(
                self.lstm.num_layers * 2,  # bidirectional
                batch_size,
                self.lstm.hidden_size
            )
            c0 = self.init_c(circuit_context).view(
                self.lstm.num_layers * 2,  # bidirectional
                batch_size,
                self.lstm.hidden_size
            )
            hidden = (h0, c0)
        else:
            hidden = None
        
        # Apply LSTM
        outputs, _ = self.lstm(x, hidden)
        
        # Apply output projection
        outputs = self.output_layer(outputs)
        
        # Apply mask if provided
        if mask is not None:
            outputs = outputs.squeeze(-1).masked_fill(~mask, 0)
        else:
            outputs = outputs.squeeze(-1)
        
        return outputs
