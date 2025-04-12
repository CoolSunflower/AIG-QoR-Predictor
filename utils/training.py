import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    """Early stopping to terminate training when validation loss doesn't improve."""
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    """Trainer class for QoR predictor model."""
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        test_loader=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        patience=10,
        clip_grad_norm=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.clip_grad_norm = clip_grad_norm
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=learning_rate / 100
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=patience,
            path=self.checkpoint_path
        )
        
        # Loss function
        self.criterion = nn.MSELoss(reduction='none')
        
        # Initialize tensorboard
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_transformer_loss = 0
        total_lstm_loss = 0
        total_samples = 0
        
        # Progress tracking
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            batch['circuit_data'] = batch['circuit_data'].to(self.device)
            batch['recipe_features'] = batch['recipe_features'].to(self.device)
            batch['recipe_edge_index'] = batch['recipe_edge_index'].to(self.device)
            batch['design_indices'] = batch['design_indices'].to(self.device)
            batch['and_counts'] = batch['and_counts'].to(self.device)
            
            if 'step_masks' in batch:
                batch['step_masks'] = batch['step_masks'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Calculate loss with masking for valid steps
            transformer_pred = outputs['transformer_pred']
            lstm_pred = outputs['lstm_pred']
            combined_pred = outputs['combined_pred']
            targets = batch['and_counts']
            
            # Apply mask if available
            if 'step_masks' in batch and batch['step_masks'] is not None:
                masks = batch['step_masks']
                transformer_loss = self.criterion(transformer_pred, targets)
                lstm_loss = self.criterion(lstm_pred, targets)
                combined_loss = self.criterion(combined_pred, targets)
                
                # Apply mask to losses
                transformer_loss = (transformer_loss * masks.float()).sum() / masks.float().sum()
                lstm_loss = (lstm_loss * masks.float()).sum() / masks.float().sum()
                combined_loss = (combined_loss * masks.float()).sum() / masks.float().sum()
                
                # Count valid samples
                num_samples = masks.float().sum().item()
            else:
                transformer_loss = self.criterion(transformer_pred, targets).mean()
                lstm_loss = self.criterion(lstm_pred, targets).mean()
                combined_loss = self.criterion(combined_pred, targets).mean()
                num_samples = targets.numel()
            
            # Total loss
            loss = combined_loss
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += combined_loss.item() * num_samples
            total_transformer_loss += transformer_loss.item() * num_samples
            total_lstm_loss += lstm_loss.item() * num_samples
            total_samples += num_samples
            
            # Log progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                elapsed = time.time() - start_time
                print(f'Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Time: {elapsed:.2f}s')
        
        # Update learning rate
        self.scheduler.step(epoch)
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        avg_transformer_loss = total_transformer_loss / total_samples if total_samples > 0 else float('inf')
        avg_lstm_loss = total_lstm_loss / total_samples if total_samples > 0 else float('inf')
        
        # Log metrics
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/TransformerLoss', avg_transformer_loss, epoch)
        self.writer.add_scalar('Train/LSTMLoss', avg_lstm_loss, epoch)
        self.writer.add_scalar('Train/LR', current_lr, epoch)
        
        return avg_loss
    
    def evaluate(self, dataloader, epoch=None, mode='val'):
        """Evaluate the model on the given dataloader."""
        self.model.eval()
        total_loss = 0
        total_transformer_loss = 0
        total_lstm_loss = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                batch['circuit_data'] = batch['circuit_data'].to(self.device)
                batch['recipe_features'] = batch['recipe_features'].to(self.device)
                batch['recipe_edge_index'] = batch['recipe_edge_index'].to(self.device)
                batch['design_indices'] = batch['design_indices'].to(self.device)
                batch['and_counts'] = batch['and_counts'].to(self.device)
                
                if 'step_masks' in batch:
                    batch['step_masks'] = batch['step_masks'].to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Get predictions and targets
                transformer_pred = outputs['transformer_pred']
                lstm_pred = outputs['lstm_pred']
                combined_pred = outputs['combined_pred']
                targets = batch['and_counts']
                
                # Calculate losses with masking for valid steps
                if 'step_masks' in batch and batch['step_masks'] is not None:
                    masks = batch['step_masks']
                    transformer_loss = self.criterion(transformer_pred, targets)
                    lstm_loss = self.criterion(lstm_pred, targets)
                    combined_loss = self.criterion(combined_pred, targets)
                    
                    # Apply mask to losses
                    transformer_loss = (transformer_loss * masks.float()).sum() / masks.float().sum()
                    lstm_loss = (lstm_loss * masks.float()).sum() / masks.float().sum()
                    combined_loss = (combined_loss * masks.float()).sum() / masks.float().sum()
                    
                    # Count valid samples
                    num_samples = masks.float().sum().item()
                    
                    # Collect predictions and targets for valid steps
                    all_preds.append(combined_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    all_masks.append(masks.cpu().numpy())
                else:
                    transformer_loss = self.criterion(transformer_pred, targets).mean()
                    lstm_loss = self.criterion(lstm_pred, targets).mean()
                    combined_loss = self.criterion(combined_pred, targets).mean()
                    num_samples = targets.numel()
                    
                    # Collect all predictions and targets
                    all_preds.append(combined_pred.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    all_masks.append(np.ones_like(targets.cpu().numpy(), dtype=bool))
                
                # Accumulate metrics
                total_loss += combined_loss.item() * num_samples
                total_transformer_loss += transformer_loss.item() * num_samples
                total_lstm_loss += lstm_loss.item() * num_samples
                total_samples += num_samples
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        avg_transformer_loss = total_transformer_loss / total_samples if total_samples > 0 else float('inf')
        avg_lstm_loss = total_lstm_loss / total_samples if total_samples > 0 else float('inf')
        
        # Calculate additional metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_masks = np.concatenate(all_masks)
        
        # Apply masks
        valid_preds = all_preds[all_masks]
        valid_targets = all_targets[all_masks]
        
        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean((valid_preds - valid_targets) ** 2))
        mae = np.mean(np.abs(valid_preds - valid_targets))
        
        # Log metrics if in validation mode
        if epoch is not None and mode == 'val':
            self.writer.add_scalar(f'{mode.capitalize()}/Loss', avg_loss, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/TransformerLoss', avg_transformer_loss, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/LSTMLoss', avg_lstm_loss, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/RMSE', rmse, epoch)
            self.writer.add_scalar(f'{mode.capitalize()}/MAE', mae, epoch)
        
        return {
            'loss': avg_loss,
            'transformer_loss': avg_transformer_loss,
            'lstm_loss': avg_lstm_loss,
            'rmse': rmse,
            'mae': mae
        }
    
    def train(self):
        """Train the model."""
        print(f"Training on {self.device}")
        print(f"Number of training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Number of validation batches: {len(self.val_loader)}")
        
        for epoch in range(1, self.max_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate on validation set
            if self.val_loader:
                val_metrics = self.evaluate(self.val_loader, epoch, mode='val')
                val_loss = val_metrics['loss']
                
                # Print epoch summary
                print(f"Epoch {epoch}/{self.max_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val RMSE: {val_metrics['rmse']:.4f} | "
                      f"Val MAE: {val_metrics['mae']:.4f}")
                
                # Check for early stopping
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            else:
                # Print epoch summary without validation
                print(f"Epoch {epoch}/{self.max_epochs} | Train Loss: {train_loss:.4f}")
        
        # Load best model for final evaluation
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            print(f"Loaded best model from {self.checkpoint_path}")
        
        # Evaluate on test set if available
        if self.test_loader:
            test_metrics = self.evaluate(self.test_loader, mode='test')
            print(f"Test Loss: {test_metrics['loss']:.4f} | "
                  f"Test RMSE: {test_metrics['rmse']:.4f} | "
                  f"Test MAE: {test_metrics['mae']:.4f}")
        
        return self.model
