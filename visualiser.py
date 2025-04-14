import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime
import networkx as nx
from tqdm import tqdm
import torch

class TrainingVisualizer:
    """
    Class for creating and saving visualizations during model training.
    """
    def __init__(self, log_dir='visualizations'):
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        self.val_final_losses = []
        self.val_step_losses = []
        self.learning_rates = []
        self.epochs = []
        
        # Create directory for visualizations
        os.makedirs(log_dir, exist_ok=True)
        
        # Subdirectories for different visualization types
        self.dirs = {
            'metrics': os.path.join(log_dir, 'metrics'),
            'predictions': os.path.join(log_dir, 'predictions'),
            'attention': os.path.join(log_dir, 'attention'),
            'circuits': os.path.join(log_dir, 'circuits')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set(font_scale=1.2)
        
    def update_metrics(self, epoch, train_loss, val_loss, val_final_loss, val_step_loss, lr):
        """Update training metrics."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_final_losses.append(val_final_loss)
        self.val_step_losses.append(val_step_loss)
        self.learning_rates.append(lr)
        
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.plot(self.epochs, self.val_final_losses, 'g--', label='Val Final Loss')
        plt.plot(self.epochs, self.val_step_losses, 'y--', label='Val Step Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs, self.learning_rates, 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], f'training_curves_epoch_{self.epochs[-1]}.png'))
        plt.close()
        
    def plot_prediction_comparison(self, true_areas, predicted_areas, step_names, epoch, prefix='val'):
        """
        Plot comparison between true and predicted areas.
        
        Args:
            true_areas: Tensor of true areas [batch_size, num_steps]
            predicted_areas: Tensor of predicted areas [batch_size, num_steps]
            step_names: List of step names
            epoch: Current epoch number
            prefix: Prefix for the plot title ('val' or 'test')
        """
        # Convert to numpy if tensors
        if torch.is_tensor(true_areas):
            true_areas = true_areas.cpu().numpy()
        if torch.is_tensor(predicted_areas):
            predicted_areas = predicted_areas.cpu().numpy()
            
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Create subplots for a subset of samples
        num_samples = min(4, true_areas.shape[0])
        for i in range(num_samples):
            plt.subplot(2, 2, i + 1)
            plt.plot(true_areas[i], 'b-o', label='True Area')
            plt.plot(predicted_areas[i], 'r-^', label='Predicted Area')
            plt.title(f'Sample {i+1}')
            plt.xlabel('Synthesis Step')
            plt.ylabel('AND Gate Count')
            plt.xticks(range(len(step_names)), step_names, rotation=45, ha='right')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['predictions'], 
                                f'{prefix}_predictions_epoch_{epoch}_samples.png'))
        plt.close()
        
        # Plot aggregated metrics
        self.plot_error_metrics(true_areas, predicted_areas, step_names, epoch, prefix)
        
    def plot_error_metrics(self, true_areas, predicted_areas, step_names, epoch, prefix='val'):
        """Plot error metrics for each synthesis step."""
        # Convert to numpy if tensors
        if torch.is_tensor(true_areas):
            true_areas = true_areas.cpu().numpy()
        if torch.is_tensor(predicted_areas):
            predicted_areas = predicted_areas.cpu().numpy()
        
        # Calculate metrics for each step
        mse_per_step = np.mean((true_areas - predicted_areas)**2, axis=0)
        mae_per_step = np.mean(np.abs(true_areas - predicted_areas), axis=0)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot MSE per step
        plt.subplot(2, 1, 1)
        plt.bar(range(len(mse_per_step)), mse_per_step)
        plt.title('Mean Squared Error per Synthesis Step')
        plt.xlabel('Synthesis Step')
        plt.ylabel('MSE')
        plt.xticks(range(len(step_names)), step_names, rotation=45, ha='right')
        plt.grid(True)
        
        # Plot MAE per step
        plt.subplot(2, 1, 2)
        plt.bar(range(len(mae_per_step)), mae_per_step)
        plt.title('Mean Absolute Error per Synthesis Step')
        plt.xlabel('Synthesis Step')
        plt.ylabel('MAE')
        plt.xticks(range(len(step_names)), step_names, rotation=45, ha='right')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], 
                                f'{prefix}_error_metrics_epoch_{epoch}.png'))
        plt.close()

        
    def plot_area_reduction(self, true_areas, predicted_areas, step_names, epoch, prefix='val'):
        """Plot area reduction through synthesis steps."""
        # Convert to numpy if tensors
        if torch.is_tensor(true_areas):
            true_areas = true_areas.cpu().numpy()
        if torch.is_tensor(predicted_areas):
            predicted_areas = predicted_areas.cpu().numpy()
            
        # Calculate relative area reduction (normalized to first step)
        true_rel_areas = true_areas / true_areas[:, 0:1]
        pred_rel_areas = predicted_areas / predicted_areas[:, 0:1]
        
        # Calculate mean and std
        true_mean = np.mean(true_rel_areas, axis=0)
        true_std = np.std(true_rel_areas, axis=0)
        pred_mean = np.mean(pred_rel_areas, axis=0)
        pred_std = np.std(pred_rel_areas, axis=0)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Plot true area reduction
        plt.plot(true_mean, 'b-', label='True Area Reduction')
        plt.fill_between(range(len(true_mean)), 
                         true_mean - true_std, 
                         true_mean + true_std, 
                         alpha=0.3, color='blue')
        
        # Plot predicted area reduction
        plt.plot(pred_mean, 'r-', label='Predicted Area Reduction')
        plt.fill_between(range(len(pred_mean)), 
                         pred_mean - pred_std, 
                         pred_mean + pred_std, 
                         alpha=0.3, color='red')
        
        plt.title('Relative Area Reduction through Synthesis Steps')
        plt.xlabel('Synthesis Step')
        plt.ylabel('Relative Area (normalized)')
        plt.xticks(range(len(step_names)), step_names, rotation=45, ha='right')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['predictions'], 
                                f'{prefix}_area_reduction_epoch_{epoch}.png'))
        plt.close()
        
    def plot_uncertainty(self, predicted_areas, uncertainties, true_areas, epoch, prefix='val'):
        """Plot predictions with uncertainty bands."""
        # Convert to numpy if tensors
        if torch.is_tensor(true_areas):
            true_areas = true_areas.cpu().numpy()
        if torch.is_tensor(predicted_areas):
            predicted_areas = predicted_areas.cpu().numpy()
        if torch.is_tensor(uncertainties):
            uncertainties = uncertainties.cpu().numpy()
            
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Create subplots for a subset of samples
        num_samples = min(4, true_areas.shape[0])
        
        for i in range(num_samples):
            plt.subplot(2, 2, i + 1)
            
            # True areas
            plt.plot(true_areas[i], 'k-o', label='True Area')
            
            # Predicted areas with uncertainty bands
            pred = predicted_areas[i]
            uncert = uncertainties[i]
            
            plt.plot(pred, 'r-^', label='Predicted Area')
            plt.fill_between(range(len(pred)), 
                             pred - uncert, 
                             pred + uncert, 
                             alpha=0.3, color='red',
                             label='Uncertainty (±σ)')
            
            plt.title(f'Sample {i+1}')
            plt.xlabel('Synthesis Step')
            plt.ylabel('AND Gate Count')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['predictions'], 
                                f'{prefix}_uncertainty_epoch_{epoch}.png'))
        plt.close()
        
    def plot_calibration(self, predicted_areas, uncertainties, true_areas, epoch, prefix='val'):
        """Plot uncertainty calibration."""
        # Convert to numpy if tensors
        if torch.is_tensor(true_areas):
            true_areas = true_areas.cpu().numpy()
        if torch.is_tensor(predicted_areas):
            predicted_areas = predicted_areas.cpu().numpy()
        if torch.is_tensor(uncertainties):
            uncertainties = uncertainties.cpu().numpy()
            
        # Flatten arrays
        pred_flat = predicted_areas.flatten()
        true_flat = true_areas.flatten()
        uncert_flat = uncertainties.flatten()
        
        # Standardized error
        z_score = np.abs(pred_flat - true_flat) / uncert_flat
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot histogram of z-scores
        plt.hist(z_score, bins=50, alpha=0.8, density=True)
        
        # Plot theoretical normal distribution
        x = np.linspace(0, 5, 1000)
        plt.plot(x, np.exp(-x**2/2) * np.sqrt(2/np.pi), 'r-', 
                linewidth=2, label='Standard half-normal')
        
        plt.title('Uncertainty Calibration')
        plt.xlabel('|Prediction Error| / Uncertainty')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['metrics'], 
                                f'{prefix}_calibration_epoch_{epoch}.png'))
        plt.close()
    
    def visualize_attention(self, attention_weights, recipe, circuit_graph, epoch, sample_idx=0):
        """
        Visualize attention weights for a sample.
        
        Args:
            attention_weights: Attention weights tensor [batch_size, seq_len, seq_len]
            recipe: Recipe commands
            circuit_graph: NetworkX graph of the circuit
            epoch: Current epoch
            sample_idx: Index of the sample to visualize
        """
        # Convert to numpy if tensor
        if torch.is_tensor(attention_weights):
            attention_weights = attention_weights[sample_idx].cpu().numpy()
        
        # Recipe attention visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, cmap='viridis', 
                   xticklabels=recipe, yticklabels=recipe)
        plt.title('Recipe Self-Attention Weights')
        plt.xlabel('Target Step')
        plt.ylabel('Source Step')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['attention'], 
                                f'recipe_attention_epoch_{epoch}_sample_{sample_idx}.png'))
        plt.close()
        
        # Circuit visualization with node importance
        if circuit_graph is not None:
            plt.figure(figsize=(12, 12))
            
            # Extract node positions using a layout algorithm
            pos = nx.spring_layout(circuit_graph, seed=42)
            
            # Draw nodes with colors based on type
            node_colors = []
            for node in circuit_graph.nodes():
                node_type = circuit_graph.nodes[node].get('type', '')
                if node_type == 'INPUT':
                    node_colors.append('lightblue')
                elif node_type == 'OUTPUT':
                    node_colors.append('lightgreen')
                elif node_type == 'AND':
                    node_colors.append('salmon')
                elif node_type == 'NOT':
                    node_colors.append('yellow')
                else:
                    node_colors.append('gray')
            
            # Calculate node sizes based on their importance (degree)
            node_sizes = [100 + 20 * circuit_graph.degree(node) for node in circuit_graph.nodes()]
            
            # Draw the graph
            nx.draw(circuit_graph, pos, 
                   node_color=node_colors,
                   node_size=node_sizes,
                   with_labels=True,
                   font_size=8,
                   edge_color='gray',
                   width=0.5,
                   arrows=True)
            
            plt.title('Circuit Structure Visualization')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['circuits'], 
                                    f'circuit_graph_epoch_{epoch}_sample_{sample_idx}.png'))
            plt.close()
