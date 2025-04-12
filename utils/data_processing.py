import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

COMMANDS = ["rewrite -z", "rewrite -l", "rewrite", "balance", "resub", "refactor", "resub -z", "refactor -z"]
CMD_TO_IDX = {cmd: i for i, cmd in enumerate(COMMANDS)}

def parse_recipe(recipe_steps):
    """
    Parse recipe steps from dataset.
    
    Args:
        recipe_steps: List of recipe step strings
        
    Returns:
        one_hot_encoding: One-hot encoding of recipe steps
        positions: Normalized positions of steps
    """
    num_steps = len(recipe_steps)
    one_hot_encoding = np.zeros((num_steps, len(COMMANDS)))
    positions = np.zeros(num_steps)
    
    for i, step in enumerate(recipe_steps):
        if step in CMD_TO_IDX:
            one_hot_encoding[i, CMD_TO_IDX[step]] = 1.0
        positions[i] = i / (num_steps - 1) if num_steps > 1 else 0
    
    return one_hot_encoding, positions

def create_recipe_graph(recipe_steps):
    """
    Create recipe graph with edges between sequential steps, 
    same type steps, and zero-cost steps.
    
    Args:
        recipe_steps: List of recipe step strings
        
    Returns:
        edge_index: Edge index for the recipe graph
    """
    num_steps = len(recipe_steps)
    edges = []
    
    # Sequential connections
    for i in range(num_steps - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))  # Bidirectional
    
    # Same type connections
    step_types = [step.split()[0] for step in recipe_steps]  # Get base command without flags
    for i in range(num_steps):
        for j in range(i + 1, num_steps):
            if step_types[i] == step_types[j]:
                edges.append((i, j))
                edges.append((j, i))  # Bidirectional
    
    # Zero-cost connections
    zero_cost_steps = [i for i, step in enumerate(recipe_steps) if "-z" in step]
    for i in range(len(zero_cost_steps)):
        for j in range(i + 1, len(zero_cost_steps)):
            edges.append((zero_cost_steps[i], zero_cost_steps[j]))
            edges.append((zero_cost_steps[j], zero_cost_steps[i]))  # Bidirectional
    
    # Remove duplicates and convert to tensor format
    edges = list(set(edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return edge_index

def load_dataset(csv_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(csv_path)
    data = []
    
    # Reference AND counts for each design from the attached image
    # These values should match the image you provided
    reference_and_counts = {
        "apex1": 1577//4,
        "bc0": 1592//4,
        "c6288": 2337//4,
        "c7582": 2198//4,
        "i2c": 1169//4,
        "max": 2865//4,
        "sasc": 613//4,
        "simple_spi": 930//4,
    }

    for _, row in df.iterrows():
        design_name = row['design_name']
        
        # Extract recipe steps and AND gate counts
        recipe_steps = []
        and_counts = []
        
        for i in range(1, 16):  # Assuming up to 15 steps
            step_key = f'Step{i}'
            and_key = f'AND{i}'
            
            if step_key in row and pd.notna(row[step_key]):
                recipe_steps.append(row[step_key])
                # Normalize AND count if requested and reference value exists
                if design_name in reference_and_counts:
                    # Normalize relative to the reference AND count for this design
                    normalized_count = row[and_key] / reference_and_counts[design_name]
                    and_counts.append(normalized_count)
                else:
                    print(f"Design {design_name} reference count not found.")
                    and_counts.append(row[and_key])        

        data.append({
            'design_name': design_name,
            'recipe_steps': recipe_steps,
            'and_counts': and_counts
        })
    
    return data

class QoRDataset(Dataset):
    """Dataset for QoR prediction."""
    def __init__(self, dataset_entries, design_data):
        self.entries = dataset_entries
        self.design_data = design_data
        self.recipe_features_cache = {}  # Cache for recipe features
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        design_name = entry['design_name']
        recipe_steps = entry['recipe_steps']
        and_counts = entry['and_counts']
        
        # Get circuit graph data
        circuit_data = self.design_data[design_name]
        
        # Process recipe
        recipe_key = tuple(recipe_steps)
        if recipe_key not in self.recipe_features_cache:
            one_hot, positions = parse_recipe(recipe_steps)
            edge_index = create_recipe_graph(recipe_steps)
            self.recipe_features_cache[recipe_key] = {
                'features': torch.tensor(np.hstack([one_hot, positions.reshape(-1, 1)]), dtype=torch.float),
                'edge_index': edge_index
            }
        
        recipe_features = self.recipe_features_cache[recipe_key]
        
        return {
            'design_name': design_name,
            'circuit_data': circuit_data,
            'recipe_features': recipe_features['features'],
            'recipe_edge_index': recipe_features['edge_index'],
            'and_counts': torch.tensor(and_counts, dtype=torch.float),
            'num_recipe_steps': len(recipe_steps)
        }

def collate_fn(batch):
    """Custom collate function for batching heterogeneous data."""
    design_names = [item['design_name'] for item in batch]
    circuit_data_list = [item['circuit_data'] for item in batch]
    recipe_features = [item['recipe_features'] for item in batch]
    recipe_edge_indices = [item['recipe_edge_index'] for item in batch]
    and_counts = [item['and_counts'] for item in batch]
    num_recipe_steps = [item['num_recipe_steps'] for item in batch]
    
    # Get unique designs
    unique_designs = list(set(design_names))
    design_to_idx = {name: i for i, name in enumerate(unique_designs)}
    design_indices = [design_to_idx[name] for name in design_names]
    
    # Batch circuit data for unique designs
    unique_circuit_data = [circuit_data_list[design_names.index(name)] for name in unique_designs]
    batched_circuit_data = Batch.from_data_list(unique_circuit_data)
    
    # Process recipe features
    max_steps = max(num_recipe_steps)
    batch_size = len(batch)
    
    # Pad recipe features
    padded_recipe_features = torch.zeros(batch_size, max_steps, recipe_features[0].size(1))
    for i, (feat, steps) in enumerate(zip(recipe_features, num_recipe_steps)):
        padded_recipe_features[i, :steps] = feat
    
    # Process recipe edge indices
    batched_recipe_edge_indices = []
    offset = 0
    for i, edge_index in enumerate(recipe_edge_indices):
        if edge_index.numel() > 0:  # Check if there are edges
            # Add offset to edge indices for batching
            edge_index_offset = edge_index.clone()
            edge_index_offset[0] += offset
            edge_index_offset[1] += offset
            batched_recipe_edge_indices.append(edge_index_offset)
        offset += num_recipe_steps[i]
    
    # Concatenate edge indices
    if batched_recipe_edge_indices:
        batched_recipe_edge_index = torch.cat(batched_recipe_edge_indices, dim=1)
    else:
        batched_recipe_edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Pad and batch and_counts
    padded_and_counts = torch.zeros(batch_size, max_steps)
    for i, (counts, steps) in enumerate(zip(and_counts, num_recipe_steps)):
        padded_and_counts[i, :steps] = counts
    
    # Create masks for valid steps
    step_masks = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    for i, steps in enumerate(num_recipe_steps):
        step_masks[i, :steps] = True
    
    return {
        'design_names': design_names,
        'unique_designs': unique_designs,
        'design_indices': torch.tensor(design_indices),
        'circuit_data': batched_circuit_data,
        'recipe_features': padded_recipe_features,
        'recipe_edge_index': batched_recipe_edge_index,
        'and_counts': padded_and_counts,
        'step_masks': step_masks,
        'num_recipe_steps': num_recipe_steps
    }
