import torch
import numpy as np
import os
import pandas as pd
from utils.circuit_parser import create_pyg_data_from_bench
from utils.data_processing import parse_recipe, create_recipe_graph


def load_model(model, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a trained model from checkpoint."""
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


def predict_single(model, design_path, recipe_steps, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make a prediction for a single design and recipe.
    
    Args:
        model: Trained QoR predictor model
        design_path: Path to the .bench file
        recipe_steps: List of recipe steps
        device: Device to use for inference
        
    Returns:
        predictions: Dictionary with step-wise predictions
    """
    model.eval()
    
    # Parse circuit
    circuit_data = create_pyg_data_from_bench(design_path)
    circuit_data = circuit_data.to(device)
    
    # Process recipe
    one_hot, positions = parse_recipe(recipe_steps)
    recipe_features = torch.tensor(
        np.hstack([one_hot, positions.reshape(-1, 1)]),
        dtype=torch.float,
        device=device
    ).unsqueeze(0)  # Add batch dimension
    
    recipe_edge_index = create_recipe_graph(recipe_steps).to(device)
    
    # Create batch dictionary
    batch = {
        'circuit_data': circuit_data.clone().to(device),
        'recipe_features': recipe_features,
        'recipe_edge_index': recipe_edge_index,
        'design_indices': torch.tensor([0], device=device),
        'step_masks': torch.ones(1, len(recipe_steps), dtype=torch.bool, device=device)
    }
    
    # Make prediction
    with torch.no_grad():
        outputs = model(batch)
    
    # Extract predictions
    transformer_pred = outputs['transformer_pred'].cpu().numpy()[0]
    lstm_pred = outputs['lstm_pred'].cpu().numpy()[0]
    combined_pred = outputs['combined_pred'].cpu().numpy()[0]
    
    # Truncate predictions to match recipe length
    transformer_pred = transformer_pred[:len(recipe_steps)]
    lstm_pred = lstm_pred[:len(recipe_steps)]
    combined_pred = combined_pred[:len(recipe_steps)]
    
    return {
        'recipe_steps': recipe_steps,
        'transformer_predictions': transformer_pred,
        'lstm_predictions': lstm_pred,
        'combined_predictions': combined_pred
    }


def predict_batch(model, design_paths, recipe_steps_list, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make predictions for a batch of designs and recipes.
    
    Args:
        model: Trained QoR predictor model
        design_paths: List of paths to .bench files
        recipe_steps_list: List of lists of recipe steps
        device: Device to use for inference
        
    Returns:
        predictions: List of dictionaries with step-wise predictions
    """
    model.eval()
    
    # Process all designs
    circuit_data_list = []
    unique_design_paths = list(set(design_paths))
    design_to_idx = {path: i for i, path in enumerate(unique_design_paths)}
    
    # Parse unique circuits
    unique_circuit_data = {}
    for path in unique_design_paths:
        unique_circuit_data[path] = create_pyg_data_from_bench(path)
    
    # Create batch data
    batch_size = len(design_paths)
    max_steps = max(len(steps) for steps in recipe_steps_list)
    
    # Prepare design indices
    design_indices = torch.tensor([design_to_idx[path] for path in design_paths], device=device)
    
    # Prepare recipe features
    all_recipe_features = []
    all_recipe_edge_indices = []
    step_masks = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)
    
    for i, recipe_steps in enumerate(recipe_steps_list):
        one_hot, positions = parse_recipe(recipe_steps)
        recipe_features = np.hstack([one_hot, positions.reshape(-1, 1)])
        
        # Pad recipe features
        padded_features = np.zeros((max_steps, recipe_features.shape[1]))
        padded_features[:len(recipe_steps)] = recipe_features
        all_recipe_features.append(padded_features)
        
        # Create recipe edge index
        edge_index = create_recipe_graph(recipe_steps)
        all_recipe_edge_indices.append(edge_index)
        
        # Set mask for valid steps
        step_masks[i, :len(recipe_steps)] = True
    
    # Convert to tensors
    recipe_features = torch.tensor(np.stack(all_recipe_features), dtype=torch.float, device=device)
    
    # Process circuit data for batching
    from torch_geometric.data import Batch
    circuit_data_list = [unique_circuit_data[path] for path in unique_design_paths]
    batched_circuit_data = Batch.from_data_list([data.to(device) for data in circuit_data_list])
    
    # Process recipe edge indices
    batched_recipe_edge_indices = []
    offset = 0
    for i, edge_index in enumerate(all_recipe_edge_indices):
        if edge_index.numel() > 0:
            edge_index_offset = edge_index.clone()
            edge_index_offset[0] += offset
            edge_index_offset[1] += offset
            batched_recipe_edge_indices.append(edge_index_offset)
        offset += len(recipe_steps_list[i])
    
    if batched_recipe_edge_indices:
        batched_recipe_edge_index = torch.cat(batched_recipe_edge_indices, dim=1).to(device)
    else:
        batched_recipe_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # Create batch dictionary
    batch = {
        'circuit_data': batched_circuit_data,
        'recipe_features': recipe_features,
        'recipe_edge_index': batched_recipe_edge_index,
        'design_indices': design_indices,
        'step_masks': step_masks
    }
    
    # Make predictions
    with torch.no_grad():
        outputs = model(batch)
    
    # Extract predictions
    transformer_preds = outputs['transformer_pred'].cpu().numpy()
    lstm_preds = outputs['lstm_pred'].cpu().numpy()
    combined_preds = outputs['combined_pred'].cpu().numpy()
    
    # Format results
    results = []
    for i, recipe_steps in enumerate(recipe_steps_list):
        # Get predictions for this sample
        transformer_pred = transformer_preds[i, :len(recipe_steps)]
        lstm_pred = lstm_preds[i, :len(recipe_steps)]
        combined_pred = combined_preds[i, :len(recipe_steps)]
        
        results.append({
            'design_path': design_paths[i],
            'recipe_steps': recipe_steps,
            'transformer_predictions': transformer_pred,
            'lstm_predictions': lstm_pred,
            'combined_predictions': combined_pred
        })
    
    return results


def evaluate_predictions(predictions, true_values):
    """
    Evaluate predictions against true values.
    
    Args:
        predictions: Array of predicted values
        true_values: Array of true values
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Calculate metrics
    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_values))
    
    # Calculate relative metrics
    mape = np.mean(np.abs((true_values - predictions) / (true_values + 1e-8))) * 100
    
    # Calculate R^2 score
    ss_total = np.sum((true_values - np.mean(true_values)) ** 2)
    ss_residual = np.sum((true_values - predictions) ** 2)
    r2 = 1 - (ss_residual / (ss_total + 1e-8))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def save_predictions(predictions, output_path):
    """
    Save predictions to a CSV file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the CSV file
    """
    # Prepare data for DataFrame
    data = []
    for pred in predictions:
        design_path = pred['design_path']
        design_name = os.path.basename(design_path).split('.')[0]
        
        for i, step in enumerate(pred['recipe_steps']):
            data.append({
                'design_name': design_name,
                'step': i + 1,
                'recipe_step': step,
                'transformer_prediction': pred['transformer_predictions'][i],
                'lstm_prediction': pred['lstm_predictions'][i],
                'combined_prediction': pred['combined_predictions'][i]
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
