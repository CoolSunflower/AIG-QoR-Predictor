import os
import re
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def parse_bench_file(file_path):
    """Parse a bench file and return a NetworkX graph."""
    graph = nx.DiGraph()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    input_nodes = []
    output_nodes = []
    gate_nodes = []
    
    # First pass: identify nodes
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
            
        if line.startswith('INPUT'):
            match = re.search(r'INPUT\((\w+)\)', line)
            if match:
                node = match.group(1)
                input_nodes.append(node)
                graph.add_node(node, type='INPUT')
        
        elif line.startswith('OUTPUT'):
            match = re.search(r'OUTPUT\((\w+)\)', line)
            if match:
                node = match.group(1)
                output_nodes.append(node)
                if node not in graph:
                    graph.add_node(node, type='OUTPUT')
                else:
                    graph.nodes[node]['type'] = 'OUTPUT'
        
        elif '=' in line:
            parts = line.split('=')
            target = parts[0].strip()
            gate_nodes.append(target)
            graph.add_node(target, type='AND')
    
    # Second pass: add edges
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '' or line.startswith('INPUT') or line.startswith('OUTPUT'):
            continue
            
        if '=' in line:
            parts = line.split('=')
            target = parts[0].strip()
            expr = parts[1].strip()
            
            if 'NOT' in expr:
                # Handle NOT gates
                match = re.search(r'NOT\((\w+)\)', expr)
                if match:
                    source = match.group(1)
                    if source not in graph:
                        graph.add_node(source, type='INTERNAL')
                    graph.add_edge(source, target, type='NOT')
            
            elif 'AND' in expr:
                # Handle AND gates
                match = re.search(r'AND\((\w+),\s*(\w+)\)', expr)
                if match:
                    source1, source2 = match.group(1), match.group(2)
                    if source1 not in graph:
                        graph.add_node(source1, type='INTERNAL')
                    if source2 not in graph:
                        graph.add_node(source2, type='INTERNAL')
                    graph.add_edge(source1, target, type='NORMAL')
                    graph.add_edge(source2, target, type='NORMAL')
    
    return graph, input_nodes, output_nodes, gate_nodes


def compute_normalized_depth(graph, output_nodes):
    """Compute normalized depth from output nodes."""
    depths = {}
    max_depth = 0
    
    for output in output_nodes:
        for node in graph.nodes():
            try:
                path_length = max([len(p) for p in nx.all_simple_paths(graph, node, output)])
                depths[node] = max(depths.get(node, 0), path_length)
                max_depth = max(max_depth, depths[node])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    
    normalized_depths = {node: depth / max_depth if max_depth > 0 else 0
                         for node, depth in depths.items()}
    
    return normalized_depths


def compute_normalized_height(graph, input_nodes):
    """Compute normalized height from input nodes."""
    # Reverse the graph to make it easier to calculate paths from inputs
    rev_graph = graph.reverse()
    heights = {}
    max_height = 0
    
    for node in graph.nodes():
        max_node_height = 0
        for input_node in input_nodes:
            try:
                paths = list(nx.all_simple_paths(rev_graph, node, input_node))
                if paths:
                    max_node_height = max(max_node_height, max(len(p) for p in paths))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        heights[node] = max_node_height
        max_height = max(max_height, max_node_height)
    
    normalized_heights = {node: height / max_height if max_height > 0 else 0
                          for node, height in heights.items()}
    
    return normalized_heights


def compute_normalized_fanout(graph):
    """Compute normalized fan-out count."""
    fanouts = {node: len(list(graph.successors(node))) for node in graph.nodes()}
    max_fanout = max(fanouts.values()) if fanouts else 1
    
    normalized_fanouts = {node: fanout / max_fanout if max_fanout > 0 else 0
                          for node, fanout in fanouts.items()}
    
    return normalized_fanouts


def compute_input_types(graph):
    """Compute one-hot encoding for input types."""
    input_types = {}
    
    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))
        if len(predecessors) == 2:  # AND gate with two inputs
            edge_types = [graph[pred][node]['type'] for pred in predecessors]
            if all(et == 'NOT' for et in edge_types):
                input_types[node] = [0, 0, 1]  # Both NOT
            elif any(et == 'NOT' for et in edge_types):
                input_types[node] = [0, 1, 0]  # One NOT, One Normal
            else:
                input_types[node] = [1, 0, 0]  # Both Normal
        else:
            input_types[node] = [0, 0, 0]  # Not applicable
    
    return input_types


def compute_input_receptive_field(graph, input_nodes):
    """Compute input receptive field: percentage of inputs affecting this node."""
    input_receptive_fields = {}
    
    for node in graph.nodes():
        affecting_inputs = 0
        for input_node in input_nodes:
            if nx.has_path(graph, input_node, node):
                affecting_inputs += 1
        
        if input_nodes:
            input_receptive_fields[node] = affecting_inputs / len(input_nodes)
        else:
            input_receptive_fields[node] = 0
    
    return input_receptive_fields


def compute_output_impact_field(graph, output_nodes):
    """Compute output impact field: percentage of outputs affected by this node."""
    output_impact_fields = {}
    
    for node in graph.nodes():
        affected_outputs = 0
        for output_node in output_nodes:
            if nx.has_path(graph, node, output_node):
                affected_outputs += 1
        
        if output_nodes:
            output_impact_fields[node] = affected_outputs / len(output_nodes)
        else:
            output_impact_fields[node] = 0
    
    return output_impact_fields


def create_node_features(graph, input_nodes, output_nodes):
    """Create initial node feature vectors."""
    normalized_depths = compute_normalized_depth(graph, output_nodes)
    normalized_heights = compute_normalized_height(graph, input_nodes)
    normalized_fanouts = compute_normalized_fanout(graph)
    input_types = compute_input_types(graph)
    input_receptive_fields = compute_input_receptive_field(graph, input_nodes)
    output_impact_fields = compute_output_impact_field(graph, output_nodes)
    
    # Node type one-hot encoding: [INPUT, OUTPUT, AND]
    node_types = {}
    for node in graph.nodes():
        if node in input_nodes:
            node_types[node] = [1, 0, 0]
        elif node in output_nodes:
            node_types[node] = [0, 1, 0]
        else:
            node_types[node] = [0, 0, 1]
    
    node_features = {}
    for node in graph.nodes():
        features = [
            normalized_depths.get(node, 0),
            normalized_heights.get(node, 0),
            normalized_fanouts.get(node, 0),
            *node_types.get(node, [0, 0, 0]),
            *input_types.get(node, [0, 0, 0]),
            input_receptive_fields.get(node, 0),
            output_impact_fields.get(node, 0)
        ]
        node_features[node] = features
    
    # Convert to node feature matrix
    nodes = list(graph.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    num_nodes = len(nodes)
    feature_dim = len(list(node_features.values())[0]) if node_features else 0
    node_feature_matrix = np.zeros((num_nodes, feature_dim))
    
    for node, features in node_features.items():
        node_feature_matrix[node_idx[node]] = features
    
    # Create edge index and edge attributes
    edges = list(graph.edges(data=True))
    edge_index = []
    edge_attr = []
    
    for u, v, data in edges:
        edge_index.append([node_idx[u], node_idx[v]])
        # Edge type: [Normal, NOT]
        if data['type'] == 'NOT':
            edge_attr.append([0, 1])
        else:
            edge_attr.append([1, 0])
    
    if edges:
        edge_index = np.array(edge_index).T
        edge_attr = np.array(edge_attr)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 2), dtype=np.float32)
    
    return node_feature_matrix, edge_index, edge_attr, node_idx


def create_pyg_data_from_bench(file_path):
    """Create PyTorch Geometric Data object from bench file."""
    graph, input_nodes, output_nodes, gate_nodes = parse_bench_file(file_path)
    node_features, edge_index, edge_attr, node_idx = create_node_features(graph, input_nodes, output_nodes)
    
    # Convert to PyG Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    
    # Save node types for reference
    data.input_nodes = [node_idx[n] for n in input_nodes]
    data.output_nodes = [node_idx[n] for n in output_nodes]
    data.gate_nodes = [node_idx[n] for n in gate_nodes]
    
    return data


def process_all_designs(design_dir):
    """Process all bench files in the design directory."""
    design_data = {}
    
    for filename in os.listdir(design_dir):
        if filename.endswith('.bench'):
            design_name = os.path.splitext(filename)[0]
            file_path = os.path.join(design_dir, filename)
            design_data[design_name] = create_pyg_data_from_bench(file_path)
    
    return design_data
