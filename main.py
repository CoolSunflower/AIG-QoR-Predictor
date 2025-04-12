import os
import torch
import numpy as np
import pandas as pd
import argparse
import random
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split

from utils.circuit_parser import process_all_designs
from utils.data_processing import load_dataset, QoRDataset, collate_fn
from models.qor_predictor import QoRPredictor
from utils.training import Trainer
from utils.inference import load_model, predict_single, predict_batch, save_predictions


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.mode == 'preprocess':
        print("Preprocessing circuit designs...")
        design_data = process_all_designs(args.design_dir)
        torch.save(design_data, args.processed_designs_path)
        print(f"Processed {len(design_data)} designs, saved to {args.processed_designs_path}")
        
    elif args.mode == 'train':
        # Load dataset
        print("Loading dataset...")
        dataset_entries = load_dataset(args.dataset_path)
        print(f"Loaded {len(dataset_entries)} dataset entries")
        
        # Load preprocessed circuit data
        print(f"Loading preprocessed circuit data from {args.processed_designs_path}")
        design_data = torch.load(args.processed_designs_path, weights_only=False)
        print(f"Loaded {len(design_data)} preprocessed designs")
        
        # Create dataset
        dataset = QoRDataset(dataset_entries, design_data)
        
        # Split dataset
        train_size = int(len(dataset) * args.train_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False
        )
        
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
            dropout=args.dropout
        ).to(device)
        
        # Print model summary
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            clip_grad_norm=args.clip_grad_norm,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Save final model
        final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
    elif args.mode == 'test':
        # Load dataset
        print("Loading test dataset...")
        dataset_entries = load_dataset(args.test_dataset_path)
        print(f"Loaded {len(dataset_entries)} test dataset entries")
        
        # Load preprocessed circuit data
        print(f"Loading preprocessed circuit data from {args.processed_designs_path}")
        design_data = torch.load(args.processed_designs_path)
        print(f"Loaded {len(design_data)} preprocessed designs")
        
        # Create dataset and dataloader
        test_dataset = QoRDataset(dataset_entries, design_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False
        )
        
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
            dropout=0.0  # No dropout during testing
        )
        
        # Load model weights
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        model = load_model(model, checkpoint_path, device)
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            train_loader=None,
            val_loader=None,
            device=device
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader, mode='test')
        
        # Print metrics
        print(f"Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(args.log_dir, 'test_metrics.csv')
        pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
        print(f"Test metrics saved to {metrics_path}")
        
    elif args.mode == 'predict':
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
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        model = load_model(model, checkpoint_path, device)
        
        # Load prediction data
        print(f"Loading prediction data from {args.predict_data_path}")
        pred_data = pd.read_csv(args.predict_data_path)
        
        # Extract design paths and recipe steps
        design_paths = []
        recipe_steps_list = []
        
        for _, row in pred_data.iterrows():
            design_name = row['design_name']
            design_path = os.path.join(args.design_dir, f"{design_name}.bench")
            
            # Extract recipe steps
            recipe_steps = []
            for i in range(1, 16):  # Assuming maximum 15 steps
                step_key = f'step{i}'
                if step_key in row and not pd.isna(row[step_key]):
                    recipe_steps.append(row[step_key])
            
            design_paths.append(design_path)
            recipe_steps_list.append(recipe_steps)
        
        # Make predictions
        print(f"Making predictions for {len(design_paths)} designs...")
        predictions = predict_batch(model, design_paths, recipe_steps_list, device)
        
        # Save predictions
        output_path = os.path.join(args.log_dir, 'predictions.csv')
        save_predictions(predictions, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QoR Prediction Model")
    
    # General parameters
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'test', 'predict'],
                        default='train', help='Mode of operation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Data paths
    parser.add_argument('--dataset_path', type=str, default='dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--design_dir', type=str, default='designs/',
                        help='Directory containing design bench files')
    parser.add_argument('--processed_designs_path', type=str, default='processed_designs.pt',
                        help='Path to save/load processed designs')
    parser.add_argument('--test_dataset_path', type=str, default='test_dataset.csv',
                        help='Path to test dataset CSV file')
    parser.add_argument('--predict_data_path', type=str, default='predict_data.csv',
                        help='Path to data for prediction')
    
    # Output paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs and results')
    
    # Model parameters
    parser.add_argument('--node_feature_dim', type=int, default=11,
                        help='Dimension of node features')
    parser.add_argument('--edge_feature_dim', type=int, default=2,
                        help='Dimension of edge features')
    parser.add_argument('--recipe_feature_dim', type=int, default=9,
                        help='Dimension of recipe features')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension size')
    parser.add_argument('--circuit_gnn_layers', type=int, default=2,
                        help='Number of GNN layers for circuit encoder')
    parser.add_argument('--recipe_gnn_layers', type=int, default=2,
                        help='Number of GNN layers for recipe encoder')
    parser.add_argument('--transformer_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/validation split ratio')
    
    args = parser.parse_args()
    main(args)
