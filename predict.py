import torch
import argparse
import os
import pandas as pd
from models.qor_predictor import QoRPredictor
from utils.inference import load_model, predict_single, predict_batch, save_predictions


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = QoRPredictor(
        node_feature_dim=args.node_feature_dim,
        edge_feature_dim=args.edge_feature_dim,
        recipe_feature_dim=args.recipe_feature_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        circuit_gnn_layers=args.circuit_gnn_layers,
        recipe_gnn_layers=args.recipe_gnn_layers,
        transformer_layers=args.transformer_layers,
        lstm_layers=args.lstm_layers,
        attention_heads=args.attention_heads,
        dropout=0.0  # No dropout during inference
    )
    
    # Load model weights
    model = load_model(model, args.checkpoint_path, device)
    print(f"Model loaded from {args.checkpoint_path}")
    
    if args.mode == 'single':
        # Get design path
        design_path = os.path.join(args.design_dir, f"{args.design_name}.bench")
        if not os.path.exists(design_path):
            raise FileNotFoundError(f"Design file not found at {design_path}")
        
        # Parse recipe steps
        recipe_steps = []
        for step in args.recipe.split(','):
            step = step.strip()
            if step:
                recipe_steps.append(step)
        
        print(f"Making prediction for design '{args.design_name}' with {len(recipe_steps)} recipe steps")
        
        # Make prediction
        result = predict_single(model, design_path, recipe_steps, device)
        
        # Print results
        print("\nPrediction Results:")
        print(f"{'Step':<5} {'Recipe Command':<15} {'Transformer':<12} {'LSTM':<12} {'Combined':<12}")
        print("-" * 60)
        
        for i, step in enumerate(result['recipe_steps']):
            print(f"{i+1:<5} {step:<15} {result['transformer_predictions'][i]:<12.2f} "
                  f"{result['lstm_predictions'][i]:<12.2f} {result['combined_predictions'][i]:<12.2f}")
        
    elif args.mode == 'batch':
        # Load prediction data
        print(f"Loading prediction data from {args.input_file}")
        
        try:
            pred_data = pd.read_csv(args.input_file)
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
        
        # Extract design paths and recipe steps
        design_paths = []
        recipe_steps_list = []
        
        for _, row in pred_data.iterrows():
            design_name = row['design_name']
            design_path = os.path.join(args.design_dir, f"{design_name}.bench")
            
            if not os.path.exists(design_path):
                print(f"Warning: Design file not found at {design_path}, skipping")
                continue
            
            # Extract recipe steps
            recipe_steps = []
            for i in range(1, 16):  # Assuming maximum 15 steps
                step_key = f'step{i}'
                if step_key in row and not pd.isna(row[step_key]):
                    recipe_steps.append(row[step_key])
            
            if not recipe_steps:
                print(f"Warning: No recipe steps found for design {design_name}, skipping")
                continue
            
            design_paths.append(design_path)
            recipe_steps_list.append(recipe_steps)
        
        # Make predictions
        print(f"Making predictions for {len(design_paths)} designs...")
        predictions = predict_batch(model, design_paths, recipe_steps_list, device)
        
        # Save predictions
        save_predictions(predictions, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QoR Prediction Inference")
    
    # General parameters
    parser.add_argument('--mode', type=str, choices=['single', 'batch'],
                        default='single', help='Prediction mode')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    # Model parameters
    parser.add_argument('--node_feature_dim', type=int, default=10,
                        help='Dimension of node features')
    parser.add_argument('--edge_feature_dim', type=int, default=2,
                        help='Dimension of edge features')
    parser.add_argument('--recipe_feature_dim', type=int, default=9,
                        help='Dimension of recipe features')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension size')
    parser.add_argument('--circuit_gnn_layers', type=int, default=3,
                        help='Number of GNN layers for circuit encoder')
    parser.add_argument('--recipe_gnn_layers', type=int, default=2,
                        help='Number of GNN layers for recipe encoder')
    parser.add_argument('--transformer_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--attention_heads', type=int, default=8,
                        help='Number of attention heads')
    
    # Model path
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--design_dir', type=str, default='designs/',
                        help='Directory containing design bench files')
    
    # Single prediction parameters
    parser.add_argument('--design_name', type=str,
                        help='Design name for single prediction (without .bench)')
    parser.add_argument('--recipe', type=str,
                        help='Comma-separated recipe steps for single prediction')
    
    # Batch prediction parameters
    parser.add_argument('--input_file', type=str,
                        help='Input CSV file for batch prediction')
    parser.add_argument('--output_file', type=str, default='predictions.csv',
                        help='Output CSV file for batch prediction results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'single' and (not args.design_name or not args.recipe):
        parser.error("--mode single requires --design_name and --recipe")
    elif args.mode == 'batch' and not args.input_file:
        parser.error("--mode batch requires --input_file")
    
    main(args)
