from torch import nn
import torch.nn.functional as F
import torch

def gaussian_nll_loss(pred, target, variance):
    """
    Negative log-likelihood loss for Gaussian distribution with predicted mean and variance.
    """
    return 0.5 * (torch.log(variance) + (pred - target)**2 / variance).mean()

class EnhancedCircuitLoss(nn.Module):
    """
    Enhanced loss function combining multiple domain-specific loss terms.
    """
    def __init__(self):
        super(EnhancedCircuitLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def relative_area_reduction_loss(self, pred, target):
        """
        Loss based on relative area reduction rather than absolute values.
        
        For circuit optimization, percentage reduction is often more important
        than absolute reduction, especially across different circuit sizes.
        """
        # Calculate relative change (step-to-step)
        pred_rel_change = (pred[:, 1:] - pred[:, :-1]) / (pred[:, :-1] + 1e-6)
        target_rel_change = (target[:, 1:] - target[:, :-1]) / (target[:, :-1] + 1e-6)
        
        # MSE on relative changes
        return F.mse_loss(pred_rel_change, target_rel_change)
    
    def critical_step_emphasis_loss(self, pred, target):
        """
        Emphasize accurate prediction of high-impact optimization steps.
        
        Steps that cause large area reductions are more important to predict
        accurately than steps with minimal impact.
        """
        # Calculate absolute area changes in target
        target_changes = torch.abs(target[:, 1:] - target[:, :-1])
        
        # Calculate prediction errors for each step
        step_errors = torch.abs(pred[:, 1:] - target[:, 1:])
        
        # Weight errors by the magnitude of target changes
        weighted_errors = step_errors * (1.0 + target_changes)
        
        return weighted_errors.mean()
    
    def sequence_dependency_loss(self, pred, target, recipe_embeddings):
        """
        Model the dependency of step effects on previous steps.
        
        The effect of a command often depends on which commands came before it.
        """
        # Extract sequential dependencies from recipe embeddings
        seq_sim = torch.bmm(recipe_embeddings, recipe_embeddings.transpose(1, 2))
        
        # Calculate step-by-step prediction errors
        step_errors = (pred - target).pow(2)
        
        # Weight errors based on sequential similarity
        batch_size, seq_len = pred.size()
        weighted_errors = torch.zeros_like(step_errors).to(pred.device)
        
        for i in range(seq_len):
            # For the first step, no dependencies
            if i == 0:
                weighted_errors[:, i] = step_errors[:, i]
                continue
                
            # For subsequent steps, calculate dependency on previous steps
            # Implementation without using batched matrix multiplication directly
            dependency_factors = torch.zeros(batch_size).to(pred.device)
            
            for j in range(i):
                # Get similarity between current step i and previous step j
                sim_ij = seq_sim[:, i, j]  # [batch_size]
                # Get error at previous step j
                err_j = step_errors[:, j]   # [batch_size]
                # Add weighted contribution
                dependency_factors += sim_ij * err_j
            
            # Apply dependency factor as multiplier to current error
            weighted_errors[:, i] = step_errors[:, i] * (1.0 + dependency_factors)
        
        return weighted_errors.mean()
    
    # def command_consistency_loss(self, pred, target, recipes):
    #     """
    #     Encourage consistent effects for the same command across similar circuits.
        
    #     Similar optimization commands should have similar effects on area.
    #     """
    #     batch_size, num_steps = pred.size()
    #     loss = 0.0
        
    #     # Find instances of the same command
    #     command_indices = {}
    #     for b in range(batch_size):
    #         for s in range(num_steps):
    #             cmd = torch.argmax(recipes[b, s]).item()
    #             if cmd not in command_indices:
    #                 command_indices[cmd] = []
    #             command_indices[cmd].append((b, s))
        
    #     # Calculate consistency loss for each command type
    #     for cmd, indices in command_indices.items():
    #         if len(indices) <= 1:
    #             continue
                
    #         # Get predictions and targets for this command
    #         cmd_preds = torch.stack([pred[b, s] for b, s in indices])
    #         cmd_targets = torch.stack([target[b, s] for b, s in indices])
            
    #         # Calculate relative effect of the command
    #         if indices[0][1] > 0:  # Not the first step
    #             cmd_pred_effects = cmd_preds / torch.stack([pred[b, s-1] for b, s in indices])
    #             cmd_target_effects = cmd_targets / torch.stack([target[b, s-1] for b, s in indices])
                
    #             # Variance of relative effects should be low
    #             pred_variance = torch.var(cmd_pred_effects)
    #             target_variance = torch.var(cmd_target_effects)
                
    #             # Penalize if prediction variance is much different from target variance
    #             loss += torch.abs(pred_variance - target_variance)
        
    #     return loss / max(1, len(command_indices))
    
    def forward(self, final_pred, step_preds, uncertainty, area_targets, recipe_embeddings, recipes):
        """
        Combine all loss terms with appropriate weights.
        """
        # Basic MSE losses
        final_loss = self.mse_loss(final_pred, area_targets[:, -1].unsqueeze(1))
        step_loss = self.mse_loss(step_preds, area_targets)
        
        # Domain-specific losses
        rel_area_loss = self.relative_area_reduction_loss(step_preds, area_targets)
        critical_step_loss = self.critical_step_emphasis_loss(step_preds, area_targets)
        seq_dep_loss = self.sequence_dependency_loss(step_preds, area_targets, recipe_embeddings)
        # cmd_consistency_loss = self.command_consistency_loss(step_preds, area_targets, recipes)
        
        # Uncertainty loss (Gaussian negative log-likelihood)
        uncertainty_loss = gaussian_nll_loss(
            final_pred,
            area_targets[:, -1].unsqueeze(1),
            uncertainty
        )
        
        # Combine all losses with weights
        total_loss = (
            1.0 * final_loss + 
            1.0 * rel_area_loss + 
            1.0 * step_loss + 
            0.3 * critical_step_loss + 
            0.2 * seq_dep_loss +
            # 0.2 * cmd_consistency_loss +
            0.3 * uncertainty_loss
        )
        
        # Return individual losses for logging
        return total_loss, {
            'final_loss': final_loss.item(),
            'step_loss': step_loss.item(),
            'rel_area_loss': rel_area_loss.item(),
            'critical_step_loss': critical_step_loss.item(),
            'seq_dep_loss': seq_dep_loss.item(),
            # 'cmd_consistency_loss': cmd_consistency_loss.item(),
            'uncertainty_loss': uncertainty_loss.item()
        }


class DomainIndependentLosses(nn.Module):
    """
    Domain-independent loss terms that improve general model performance.
    """
    def __init__(self):
        super(DomainIndependentLosses, self).__init__()
    
    # def gradient_penalty_loss(self, model, circuit_data, recipe_data, device):
    #     """
    #     Gradient penalty for Lipschitz continuity (improves stability).
    #     Penalizes large gradients to make the model more robust to small input changes.
    #     """
    #     batch_size = recipe_data.size(0)
        
    #     # Create interpolated inputs
    #     alpha = torch.rand(batch_size, 1, 1, device=device)
        
    #     # Perturb recipe data slightly
    #     perturbed_recipe = recipe_data.clone().detach() + 0.01 * torch.randn_like(recipe_data)
    #     interpolated_recipe = alpha * recipe_data + (1 - alpha) * perturbed_recipe
    #     interpolated_recipe.requires_grad_(True)
        
    #     # Forward pass with interpolated inputs
    #     final_pred, _, _ = model(circuit_data, interpolated_recipe)
        
    #     # Calculate gradients
    #     gradients = torch.autograd.grad(
    #         outputs=final_pred.sum(),
    #         inputs=interpolated_recipe,
    #         create_graph=True,
    #         retain_graph=True
    #     )[0]  # Extract the gradient tensor from the tuple
        
    #     # Calculate gradient penalty
    #     gradients = gradients.reshape(batch_size, -1)  # Use reshape instead of view
    #     gradient_norm = gradients.norm(2, dim=1)
    #     gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
    #     return gradient_penalty

    
    def attention_entropy_loss(self, attention_weights):
        """
        Encourages more focused (less uniform) attention patterns.
        
        Attention should be concentrated on relevant steps rather than dispersed.
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + epsilon), dim=-1)
        
        return entropy.mean()
    
    def feature_consistency_loss(self, circuit_embeddings, area_targets):
        """
        Encourages similar circuits to have similar embeddings.
        
        Circuits with similar area profiles should have similar embeddings.
        """
        batch_size = circuit_embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=circuit_embeddings.device)
        
        # Calculate pairwise embedding distances
        embedding_dists = torch.cdist(circuit_embeddings, circuit_embeddings, p=2)
        
        # Calculate pairwise area profile distances
        area_dists = torch.cdist(area_targets, area_targets, p=2)
        
        # Normalize distances
        area_dists = area_dists / (area_dists.max() + 1e-8)
        
        # Correlation between embedding distance and area distance
        # Similar area profiles should have similar embeddings
        consistency_loss = F.mse_loss(embedding_dists, area_dists)
        
        return consistency_loss
    
    def forward(self, model, circuit_data, recipe_data, circuit_embeddings, 
                attention_weights, area_targets, device):
        """
        Combine all domain-independent loss terms.
        """
        # gradient_penalty = self.gradient_penalty_loss(model, circuit_data, recipe_data, device)
        attention_entropy = self.attention_entropy_loss(attention_weights)
        feature_consistency = self.feature_consistency_loss(circuit_embeddings, area_targets)
        
        # Combine losses with weights
        total_loss = (
            # 0.1 * gradient_penalty + 
            0.1 * attention_entropy + 
            0.2 * feature_consistency
        )
        
        # Return individual losses for logging
        return total_loss, {
            # 'gradient_penalty': gradient_penalty.item(),
            'attention_entropy': attention_entropy.item(),
            'feature_consistency': feature_consistency.item()
        }
