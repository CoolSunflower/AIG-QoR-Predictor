import os
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
from visualiser import TrainingVisualizer
from losses import DomainIndependentLosses, EnhancedCircuitLoss
from tqdm import tqdm
from collections import defaultdict

BASE_AREAS = {
    'apex1': 1577,
    'bc0': 1592,
    'c6288': 2337,
    'c7552': 2198,
    'i2c': 1169,
    'max': 2865,
    'sasc': 613,
    'simple_spi': 930,
}

def parse_bench_file(file_path):
    """
    Parse a BENCH file into a NetworkX directed graph.
    
    Args:
        file_path (str): Path to the BENCH file
        
    Returns:
        nx.DiGraph: Directed graph representing the circuit
    """
    G = nx.DiGraph()
    inputs = []
    outputs = []
    gates = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Parse input
            if line.startswith('INPUT('):
                node_name = re.search(r'INPUT\((.*?)\)', line).group(1)
                G.add_node(node_name, type='INPUT', level=0)
                inputs.append(node_name)
                
            # Parse output
            elif line.startswith('OUTPUT('):
                node_name = re.search(r'OUTPUT\((.*?)\)', line).group(1)
                G.add_node(node_name, type='OUTPUT')
                outputs.append(node_name)
                
            # Parse gates
            elif '=' in line:
                parts = line.split('=', 1)
                target = parts[0].strip()
                expression = parts[1].strip()
                
                # Determine gate type
                if 'AND(' in expression:
                    gate_type = 'AND'
                    inputs_str = re.search(r'AND\((.*?)\)', expression).group(1)
                    input_nodes = [inp.strip() for inp in inputs_str.split(',')]
                elif 'NOT(' in expression:
                    gate_type = 'NOT'
                    inputs_str = re.search(r'NOT\((.*?)\)', expression).group(1)
                    input_nodes = [inputs_str.strip()]
                else:
                    continue  # Skip unsupported gate types
                
                # Add node and edges
                G.add_node(target, type=gate_type)
                for input_node in input_nodes:
                    G.add_edge(input_node, target)
                gates[target] = {'type': gate_type, 'inputs': input_nodes}
    
    # Set node levels (topological depth)
    levels = {}
    for node in nx.topological_sort(G):
        if G.nodes[node]['type'] == 'INPUT':
            levels[node] = 0
        else:
            pred_levels = [levels.get(pred, 0) for pred in G.predecessors(node)]
            levels[node] = max(pred_levels) + 1 if pred_levels else 0
            G.nodes[node]['level'] = levels[node]
    
    # Calculate fanout for each node
    for node in G.nodes():
        G.nodes[node]['fanout'] = G.out_degree(node)
        G.nodes[node]['fanin'] = G.in_degree(node)
    
    return G

def extract_circuit_features(G):
    """
    Extract node and edge features from the circuit graph.
    
    Args:
        G (nx.DiGraph): Circuit graph
        
    Returns:
        torch_geometric.data.Data: PyTorch Geometric data object
    """
    # Node type one-hot encoding: [INPUT, OUTPUT, AND, NOT]
    node_types = {'INPUT': 0, 'OUTPUT': 1, 'AND': 2, 'NOT': 3}
    
    # Extract node features: [type, level, fanin, fanout]
    node_features = []
    for node in G.nodes():
        node_data = G.nodes[node]
        
        # One-hot encoding for node type
        node_type = [0, 0, 0, 0]
        node_type[node_types[node_data['type']]] = 1
        
        # Other features
        level = node_data.get('level', 0)
        fanin = node_data.get('fanin', 0)
        fanout = node_data.get('fanout', 0)
        
        # Combine features
        features = node_type + [level, fanin, fanout]
        node_features.append(features)
    
    # Create mapping from node names to indices
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    # Extract edges
    edge_index = []
    for source, target in G.edges():
        edge_index.append([node_mapping[source], node_mapping[target]])
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Add global graph features
    data.num_nodes = G.number_of_nodes()
    data.num_edges = G.number_of_edges()
    data.num_inputs = sum(1 for _, attr in G.nodes(data=True) if attr['type'] == 'INPUT')
    data.num_outputs = sum(1 for _, attr in G.nodes(data=True) if attr['type'] == 'OUTPUT')
    
    return data

def encode_recipe(recipe, max_steps=20):
    """
    Encode a synthesis recipe as a sequence of one-hot vectors.
    
    Args:
        recipe (list): List of synthesis commands
        max_steps (int): Maximum number of steps in the recipe
        
    Returns:
        torch.Tensor: Encoded recipe (max_steps x num_commands)
    """
    COMMANDS = ["rewrite -z", "rewrite -l", "rewrite", "balance", 
                "resub", "refactor", "resub -z", "refactor -z"]
    
    cmd_to_idx = {cmd: i for i, cmd in enumerate(COMMANDS)}
    
    # Create one-hot encoding for each command
    encoded_recipe = torch.zeros(max_steps, len(COMMANDS))
    
    for i, cmd in enumerate(recipe[:max_steps]):
        if cmd in cmd_to_idx:
            encoded_recipe[i, cmd_to_idx[cmd]] = 1.0
    
    return encoded_recipe

class CircuitDataset(Dataset):
    """
    Dataset for circuit designs and recipes.
    """
    def __init__(self, csv_path, designs_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the dataset CSV file
            designs_dir (str): Directory containing the BENCH files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = pd.read_csv(csv_path)
        self.designs_dir = designs_dir
        self.transform = transform
        
        # Extract recipe steps and area values
        self.recipes = []
        self.areas = []

        for _, row in self.data.iterrows():
            recipe = []
            areas = []
            
            for i in range(1, 21):  # 15 steps
                step_col = f'Step{i}'
                area_col = f'AND{i}'
                
                if step_col in row and area_col in row:
                    recipe.append(row[step_col])
                    areas.append(row[area_col]/(BASE_AREAS[row['design_name']]//2))
            
            self.recipes.append(recipe)
            self.areas.append(areas)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        design_name = self.data.iloc[idx]['design_name']
        bench_path = os.path.join(self.designs_dir, f"{design_name}.bench")
        
        # Parse circuit and extract features
        G = parse_bench_file(bench_path)
        circuit_data = extract_circuit_features(G)
        
        # Encode recipe
        recipe = self.recipes[idx]
        encoded_recipe = encode_recipe(recipe)
        
        # Get area values
        areas = torch.tensor(self.areas[idx], dtype=torch.float)
        
        # Combine into a sample
        sample = {
            'circuit': circuit_data,
            'recipe': encoded_recipe,
            'areas': areas,
            'design_name': design_name
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GraphAttention(nn.Module):
    """
    Graph attention module to focus on important nodes.
    """
    def __init__(self, hidden_dim):
        super(GraphAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, edge_index):
        # Calculate attention weights
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=0)
        
        # Apply attention weights
        x_weighted = x * attn_weights
        
        return x_weighted

class CircuitEmbedding(nn.Module):
    """
    GNN-based module for embedding circuit designs.
    """
    def __init__(self, node_feature_dim, hidden_dim=128, num_layers=3):
        super(CircuitEmbedding, self).__init__()
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(node_feature_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Graph attention for focusing on important nodes
        self.attention = GraphAttention(hidden_dim)
        
        # Batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Global pooling to get graph-level representation
        self.pool = global_mean_pool
        
        # Final projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Apply graph convolutions with residual connections
        for i, conv in enumerate(self.conv_layers):
            identity = x
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # Add residual connection if dimensions match
            if i > 0 and x.size(-1) == identity.size(-1):
                x = x + identity
        
        # Apply attention
        x = self.attention(x, edge_index)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Final projection
        x = self.project(x)
        
        return x

class RecipeProcessor(nn.Module):
    """
    RNN-based module for processing synthesis recipes with exposed attention weights.
    """
    def __init__(self, recipe_dim, hidden_dim=128, num_layers=2):
        super(RecipeProcessor, self).__init__()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=recipe_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Project bidirectional output to hidden_dim
        self.project = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Attention mechanism for commands
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Store attention weights for visualization
        self.last_attention_weights = None
        
    def forward(self, x):
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)
        
        # Project bidirectional output
        lstm_out = self.project(lstm_out)
        
        # Apply self-attention
        attn_out, attn_weights = self.attention(
            lstm_out.permute(1, 0, 2),  # [seq_len, batch, hidden]
            lstm_out.permute(1, 0, 2),
            lstm_out.permute(1, 0, 2)
        )
        
        # Store attention weights for visualization
        self.last_attention_weights = attn_weights
        
        # Return to original shape [batch, seq_len, hidden]
        attn_out = attn_out.permute(1, 0, 2)
        
        return lstm_out, attn_out



class StepwisePredictor(nn.Module):
    """
    Module for predicting area after each synthesis step.
    """
    def __init__(self, circuit_dim, recipe_dim, hidden_dim=128):
        super(StepwisePredictor, self).__init__()
        
        # Combine circuit and recipe representations
        self.combine = nn.Linear(circuit_dim + recipe_dim, hidden_dim)
        
        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, circuit_embedding, recipe_state):
        # Combine embeddings
        combined = torch.cat([circuit_embedding, recipe_state], dim=1)
        
        # Pass through MLP
        x = F.relu(self.combine(combined))
        area_pred = self.mlp(x)
        
        return area_pred
    

class CircuitQoRModel(nn.Module):
    """
    Complete model for predicting circuit QoR with exposed embeddings and attention.
    """
    def __init__(self, node_feature_dim, recipe_dim, hidden_dim=128):
        super(CircuitQoRModel, self).__init__()
        
        # Circuit embedding module
        self.circuit_embedder = CircuitEmbedding(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim
        )
        
        # Recipe processing module
        self.recipe_processor = RecipeProcessor(
            recipe_dim=recipe_dim,
            hidden_dim=hidden_dim
        )
        
        # Stepwise prediction module
        self.step_predictor = StepwisePredictor(
            circuit_dim=hidden_dim,
            recipe_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # Final prediction combining circuit and recipe information
        self.final_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Store last attention weights and embeddings
        self.last_attention_weights = None
        self.last_circuit_embedding = None
        self.last_recipe_embeddings = None
        
    def forward(self, circuit_data, recipe):
        # Extract circuit features
        x, edge_index = circuit_data.x, circuit_data.edge_index
        batch = circuit_data.batch if hasattr(circuit_data, 'batch') else None
        
        # Get circuit embedding
        circuit_embedding = self.circuit_embedder(x, edge_index, batch)
        self.last_circuit_embedding = circuit_embedding
        
        # Process recipe
        lstm_out, attn_out = self.recipe_processor(recipe)
        self.last_recipe_embeddings = lstm_out
        
        # Store attention weights
        if hasattr(self.recipe_processor, 'last_attention_weights'):
            self.last_attention_weights = self.recipe_processor.last_attention_weights
        
        # Stepwise prediction
        step_predictions = []
        for t in range(lstm_out.size(1)):
            step_pred = self.step_predictor(
                circuit_embedding,
                lstm_out[:, t, :]
            )
            step_predictions.append(step_pred)
        
        # Stack step predictions
        step_predictions = torch.cat(step_predictions, dim=1)
        
        # Final prediction using circuit embedding and final recipe state
        final_embedding = torch.cat([
            circuit_embedding,
            attn_out[:, -1, :]  # Use the last state with attention
        ], dim=1)
        
        final_pred = self.final_predictor(final_embedding)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty(final_embedding)
        
        return final_pred, step_predictions, uncertainty
    
    def get_attention_weights(self):
        """Return the attention weights from the recipe processor."""
        return self.last_attention_weights
    
    def get_circuit_embedding(self):
        """Return the circuit embedding."""
        return self.last_circuit_embedding
    
    def get_recipe_embeddings(self):
        """Return the recipe embeddings."""
        return self.last_recipe_embeddings


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

def collate_fn(batch):
    """
    Custom collate function for batching circuit graphs and recipes.
    """
    circuits = [item['circuit'] for item in batch]
    recipes = [item['recipe'] for item in batch]
    areas = [item['areas'] for item in batch]
    design_names = [item['design_name'] for item in batch]
    
    # Batch circuit graphs
    batched_circuits = Batch.from_data_list(circuits)
    
    # Stack recipes and areas
    batched_recipes = torch.stack(recipes, dim=0)
    batched_areas = torch.stack(areas, dim=0)
    
    return {
        'circuit': batched_circuits,
        'recipe': batched_recipes,
        'areas': batched_areas,
        'design_name': design_names
    }

def gaussian_nll_loss(pred, target, variance):
    """
    Negative log-likelihood loss for Gaussian distribution with predicted mean and variance.
    """
    return 0.5 * (torch.log(variance) + (pred - target)**2 / variance).mean()

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=1e-5,
    checkpoint_dir='checkpoints',
    log_dir='logs',
    patience=15,
    vis_dir='visualizations',
    design_name='bc0'
):
    """
    Train the QoR prediction model with enhanced loss functions.
    """
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize visualization helper
    visualizer = TrainingVisualizer(vis_dir)
    
    # Initialize domain-specific and domain-independent loss functions
    domain_loss = EnhancedCircuitLoss()
    indep_loss = DomainIndependentLosses()
    
    # List of step names for visualizations
    step_names = [f'Step {i+1}' for i in range(20)]  # Assuming 20 steps
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_epochs
    )

    # Move model to device
    model = model.to(device)
    
    # Training loop variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_breakdown = defaultdict(float)
        
        start_time = time.time()
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_progress:
            # Move data to device
            circuit_data = batch['circuit'].to(device)
            recipe_data = batch['recipe'].to(device)
            area_targets = batch['areas'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.backends.cudnn.flags(enabled=False):
                final_pred, step_preds, uncertainty = model(circuit_data, recipe_data)
            
            # Get embeddings and attention weights
            circuit_embedding = model.get_circuit_embedding()
            recipe_embeddings = model.get_recipe_embeddings()
            attention_weights = model.get_attention_weights()
            
            # Calculate domain-specific losses
            domain_total_loss, domain_losses = domain_loss(
                final_pred, 
                step_preds, 
                uncertainty, 
                area_targets, 
                recipe_embeddings,
                recipe_data
            )
            
            # Calculate domain-independent losses
            indep_total_loss, indep_losses = indep_loss(
                model,
                circuit_data,
                recipe_data,
                circuit_embedding,
                attention_weights,
                area_targets,
                device
            )
            
            # Combined loss
            loss = domain_total_loss + indep_total_loss
            
            # Backward pass and optimization
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update running loss
            epoch_loss += loss.item()
            
            # Update loss breakdown
            for k, v in domain_losses.items():
                epoch_loss_breakdown[k] += v
            for k, v in indep_losses.items():
                epoch_loss_breakdown[k] += v
            
            # Update progress bar
            train_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'domain_loss': f'{domain_total_loss.item():.4f}',
                'indep_loss': f'{indep_total_loss.item():.4f}'
            })
            
            # Log batch metrics
            writer.add_scalar('Batch/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Batch/Domain_Loss', domain_total_loss.item(), global_step)
            writer.add_scalar('Batch/Independent_Loss', indep_total_loss.item(), global_step)
            
            for k, v in domain_losses.items():
                writer.add_scalar(f'Batch/Domain/{k}', v, global_step)
            for k, v in indep_losses.items():
                writer.add_scalar(f'Batch/Independent/{k}', v, global_step)
            
            global_step += 1
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        for k in epoch_loss_breakdown:
            epoch_loss_breakdown[k] /= len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_breakdown = defaultdict(float)
        
        # Collect predictions for visualization
        all_val_preds = []
        all_val_targets = []
        all_val_uncertainties = []
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_progress:
                # Move data to device
                circuit_data = batch['circuit'].to(device)
                recipe_data = batch['recipe'].to(device)
                area_targets = batch['areas'].to(device)
                
                # Forward pass
                with torch.backends.cudnn.flags(enabled=False):
                    final_pred, step_preds, uncertainty = model(circuit_data, recipe_data)
                
                # Get embeddings and attention weights
                circuit_embedding = model.get_circuit_embedding()
                recipe_embeddings = model.get_recipe_embeddings()
                attention_weights = model.get_attention_weights()
                
                # Calculate domain-specific losses
                domain_total_loss, domain_losses = domain_loss(
                    final_pred, 
                    step_preds, 
                    uncertainty, 
                    area_targets, 
                    recipe_embeddings,
                    recipe_data
                )
                
                # Calculate domain-independent losses
                indep_total_loss, indep_losses = indep_loss(
                    model,
                    circuit_data,
                    recipe_data,
                    circuit_embedding,
                    attention_weights,
                    area_targets,
                    device
                )
                
                # Combined loss
                loss = domain_total_loss + indep_total_loss
                
                # Update validation losses
                val_loss += loss.item()
                
                # Update validation loss breakdown
                for k, v in domain_losses.items():
                    val_loss_breakdown[k] += v
                for k, v in indep_losses.items():
                    val_loss_breakdown[k] += v
                
                # Collect predictions for visualization
                all_val_preds.append(step_preds.cpu())
                all_val_targets.append(area_targets.cpu())
                all_val_uncertainties.append(uncertainty.cpu().expand_as(step_preds).cpu())
        
        # Calculate average validation losses
        val_loss /= len(val_loader)
        for k in val_loss_breakdown:
            val_loss_breakdown[k] /= len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch metrics
        writer.add_scalar('Epoch/Train_Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
        
        for k, v in epoch_loss_breakdown.items():
            writer.add_scalar(f'Epoch/Train/{k}', v, epoch)
        for k, v in val_loss_breakdown.items():
            writer.add_scalar(f'Epoch/Val/{k}', v, epoch)
        
        # Update and create visualizations
        visualizer.update_metrics(epoch, epoch_loss, val_loss, 
                                 val_loss_breakdown.get('final_loss', 0), 
                                 val_loss_breakdown.get('step_loss', 0), 
                                 current_lr)
        visualizer.plot_training_curves()
        
        # Concatenate validation predictions and targets
        if all_val_preds:
            val_preds = torch.cat(all_val_preds, dim=0)
            val_targets = torch.cat(all_val_targets, dim=0)
            val_uncertainties = torch.cat(all_val_uncertainties, dim=0)
            
            # Create visualizations
            visualizer.plot_prediction_comparison(val_targets, val_preds, step_names, epoch)
            visualizer.plot_error_metrics(val_targets, val_preds, step_names, epoch)
            visualizer.plot_area_reduction(val_targets, val_preds, step_names, epoch)
            visualizer.plot_uncertainty(val_preds, val_uncertainties, val_targets, epoch)
            visualizer.plot_calibration(val_preds, val_uncertainties, val_targets, epoch)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print("Loss breakdown:")
        for k, v in sorted(epoch_loss_breakdown.items()):
            print(f"  Train {k}: {v:.4f}, Val {k}: {val_loss_breakdown.get(k, 0):.4f}")
        print(f"Learning Rate: {current_lr}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_{design_name}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'loss_breakdown': val_loss_breakdown
        }, checkpoint_path)
        
        print(f"Model saved to {checkpoint_path}")

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve. Early stop counter: {early_stop_counter}/{patience}")
        
        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Close tensorboard writer
    writer.close()
    
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1-early_stop_counter}.pt")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def main(design_name):
    """
    Main function for training the QoR prediction model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset paths
    csv_path = f'./datasets/{design_name}.csv'
    designs_dir = 'designs'
    
    # Create dataset
    dataset = CircuitDataset(csv_path, designs_dir)
    
    # Split dataset into train, validation, and test sets
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Model hyperparameters
    node_feature_dim = 7  # 4 (one-hot node type) + 3 (level, fanin, fanout)
    recipe_dim = 8  # Number of commands
    hidden_dim = 128
    
    # Create model
    model = CircuitQoRModel(
        node_feature_dim=node_feature_dim,
        recipe_dim=recipe_dim,
        hidden_dim=hidden_dim
    )
    
    # Train model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=75,
        learning_rate=0.001,
        weight_decay=1e-5,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        patience=15,
        design_name=design_name
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), 'circuit_qor_model.pt')
    
    # Evaluate on test set
    evaluate_model(trained_model, test_loader, device)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def evaluate_model(model, test_loader, device, vis_dir = 'graphs/'):
    """
    Evaluate the model on the test set with detailed visualizations.
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Device to use for inference
        vis_dir: Directory for test visualizations
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    test_loss = 0.0
    final_loss = 0.0
    step_loss = 0.0
    mse_loss = nn.MSELoss()
    
    # Initialize visualization helper
    visualizer = TrainingVisualizer(vis_dir)
    
    # List of step names for visualizations
    step_names = [f'Step {i+1}' for i in range(20)]  # Assuming 20 steps
    
    # Collect predictions for visualization
    all_preds = []
    all_targets = []
    all_uncertainties = []
    all_design_names = []
    
    test_progress = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in test_progress:
            # Move data to device
            circuit_data = batch['circuit'].to(device)
            recipe_data = batch['recipe'].to(device)
            area_targets = batch['areas'].to(device)
            design_names = batch['design_name']
            
            # Forward pass
            final_pred, step_preds, uncertainty = model(circuit_data, recipe_data)
            
            # Calculate losses
            final_batch_loss = mse_loss(final_pred, area_targets[:, -1].unsqueeze(1))
            step_batch_loss = mse_loss(step_preds, area_targets)
            
            # Update test losses
            final_loss += final_batch_loss.item()
            step_loss += step_batch_loss.item()
            test_loss += (final_batch_loss.item() + 0.5 * step_batch_loss.item())
            
            # Store predictions and targets
            all_preds.append(step_preds.cpu())
            all_targets.append(area_targets.cpu())
            all_uncertainties.append(uncertainty.cpu().expand_as(step_preds).cpu())
            all_design_names.extend(design_names)
            
            # Update progress bar
            test_progress.set_postfix({
                'loss': f'{test_loss/len(all_preds):.4f}',
                'final_loss': f'{final_loss/len(all_preds):.4f}',
                'step_loss': f'{step_loss/len(all_preds):.4f}'
            })
    
    # Calculate average test losses
    test_loss /= len(test_loader)
    final_loss /= len(test_loader)
    step_loss /= len(test_loader)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_uncertainties = torch.cat(all_uncertainties, dim=0).numpy()
    
    # Calculate metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # R² score for each step
    r2_scores = []
    for i in range(all_preds.shape[1]):
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        r2_scores.append(r2)
    
    # Create visualizations
    visualizer.plot_prediction_comparison(all_targets, all_preds, step_names, 0, prefix='test')
    visualizer.plot_error_metrics(all_targets, all_preds, step_names, 0, prefix='test')
    visualizer.plot_area_reduction(all_targets, all_preds, step_names, 0, prefix='test')
    visualizer.plot_uncertainty(all_preds, all_uncertainties, all_targets, 0, prefix='test')
    visualizer.plot_calibration(all_preds, all_uncertainties, all_targets, 0, prefix='test')
    
    # Create detailed per-design visualizations
    detailed_results_dir = os.path.join(vis_dir, 'detailed_results')
    os.makedirs(detailed_results_dir, exist_ok=True)
    
    # Create CSV with detailed results
    results_df = pd.DataFrame({
        'design_name': all_design_names
    })
    
    for i in range(all_preds.shape[1]):
        results_df[f'true_area_{i+1}'] = all_targets[:, i]
        results_df[f'pred_area_{i+1}'] = all_preds[:, i]
        results_df[f'error_{i+1}'] = all_targets[:, i] - all_preds[:, i]
        results_df[f'abs_error_{i+1}'] = np.abs(all_targets[:, i] - all_preds[:, i])
    
    results_df['mean_abs_error'] = results_df[[f'abs_error_{i+1}' for i in range(all_preds.shape[1])]].mean(axis=1)
    results_df.to_csv(os.path.join(detailed_results_dir, 'detailed_results.csv'), index=False)
    
    # Create per-step error distribution plots
    for i in range(all_preds.shape[1]):
        plt.figure(figsize=(12, 8))
        errors = all_targets[:, i] - all_preds[:, i]
        plt.hist(errors, bins=50, alpha=0.8)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'Error Distribution for Step {i+1}')
        plt.xlabel('Prediction Error (True - Predicted)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(detailed_results_dir, f'error_distribution_step_{i+1}.png'))
        plt.close()
    
    # Create summary visualization for all steps
    plt.figure(figsize=(15, 10))
    
    # Plot R² per step
    plt.subplot(2, 1, 1)
    plt.bar(range(len(r2_scores)), r2_scores)
    plt.axhline(y=0.9, color='r', linestyle='--', label='R²=0.9')
    plt.title('R² Score per Synthesis Step')
    plt.xlabel('Synthesis Step')
    plt.ylabel('R² Score')
    plt.xticks(range(len(step_names)), step_names, rotation=45, ha='right')
    plt.grid(True)
    plt.legend()
    
    # Plot error metrics per step
    mse_per_step = np.mean((all_preds - all_targets)**2, axis=0)
    mae_per_step = np.mean(np.abs(all_preds - all_targets), axis=0)
    rmse_per_step = np.sqrt(mse_per_step)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(len(mse_per_step)), rmse_per_step, 'b-o', label='RMSE')
    plt.plot(range(len(mae_per_step)), mae_per_step, 'g-^', label='MAE')
    plt.title('Error Metrics per Synthesis Step')
    plt.xlabel('Synthesis Step')
    plt.ylabel('Error')
    plt.xticks(range(len(step_names)), step_names, rotation=45, ha='right')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'test_summary_metrics.png'))
    plt.close()
    
    # Print test results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Final Loss: {final_loss:.4f}")
    print(f"Test Step Loss: {step_loss:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Mean R² Score: {np.mean(r2_scores):.4f}")
    
    # Return metrics
    return {
        'test_loss': test_loss,
        'final_loss': final_loss,
        'step_loss': step_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_scores': r2_scores,
        'mean_r2': np.mean(r2_scores)
    }

def predict_area(circuit_bench_path, recipe, model_path=None, device=None):
    """
    Predict area for a given circuit and synthesis recipe.
    
    Args:
        circuit_bench_path (str): Path to the BENCH file
        recipe (list): List of synthesis commands
        model_path (str): Path to the trained model
        device (torch.device): Device to use for inference
        
    Returns:
        dict: {
            'final_area': Predicted final area,
            'step_areas': List of predicted areas after each step,
            'step_deltas': Area change at each step,
            'confidence': Confidence in prediction (1-σ)
        }
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if path is provided
    model = None
    if model_path is not None:
        # Create model instance
        node_feature_dim = 7
        recipe_dim = 8
        hidden_dim = 128
        
        model = CircuitQoRModel(
            node_feature_dim=node_feature_dim,
            recipe_dim=recipe_dim,
            hidden_dim=hidden_dim
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        # Load default model
        model = CircuitQoRModel(
            node_feature_dim=7,
            recipe_dim=8,
            hidden_dim=128
        )
        model.load_state_dict(torch.load('circuit_qor_model.pt', map_location=device))
        model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Parse circuit and extract features
    G = parse_bench_file(circuit_bench_path)
    circuit_data = extract_circuit_features(G)
    
    # Encode recipe
    encoded_recipe = encode_recipe(recipe)
    
    # Move data to device
    circuit_data = circuit_data.to(device)
    encoded_recipe = encoded_recipe.unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        final_area, step_areas, uncertainty = model(circuit_data, encoded_recipe)
    
    # Convert to numpy arrays
    final_area = final_area.cpu().item()
    step_areas = step_areas.squeeze(0).cpu().numpy().tolist()
    uncertainty = uncertainty.cpu().item()
    
    # Calculate deltas
    step_deltas = [step_areas[i] - step_areas[i-1] if i > 0 else 0 
                   for i in range(len(step_areas))]
    
    # Calculate confidence (1-σ)
    confidence = 1.0 - min(1.0, uncertainty)
    
    return {
        'final_area': final_area,
        'step_areas': step_areas,
        'step_deltas': step_deltas,
        'confidence': confidence
    }

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
