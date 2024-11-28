# Encodings and so on
import scipy.sparse as sp # type: ignore
from scipy.sparse.csgraph import laplacian # type: ignore
from scipy.sparse.linalg import eigsh # type: ignore
import torch # type: ignore
import numpy as np # type: ignore
import networkx as nx # type: ignore

def compute_rwse(graph, steps=5):
    """
    Compute Random Walk Structural Encoding (RWSE).
    """
    adj_matrix = nx.adjacency_matrix(graph).toarray()
    # Normalize adjacency matrix
    rw_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    rw_steps = [np.linalg.matrix_power(rw_matrix, s) for s in range(1, steps + 1)]
    rw_diag = [np.diag(rw) for rw in rw_steps]
    return np.stack(rw_diag, axis=-1)

def extend_data_w_rwse(orig_data, graph, steps = 5):
    rwse_data = compute_rwse(graph, steps=steps)
    rwse_tensor = torch.tensor(rwse_data, dtype=torch.float)
    orig_data.x = torch.cat([orig_data.x, rwse_tensor], dim=1)
    return orig_data

def add_node_degree_features(pyg_data, graph):
    """
    Add node degree as a feature to the PyG Data object.
    """
    degrees = np.array([graph.degree[node] for node in graph.nodes])
    degree_features = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
    pyg_data.x = torch.cat([pyg_data.x, degree_features], dim=1)
    return pyg_data

def compute_laplacian_pe(graph, k):
    """
    Compute Laplacian Positional Encoding (PE).
    """
    adj_matrix = nx.adjacency_matrix(graph)
    lap = laplacian(adj_matrix, normed=True)
    eigvals, eigvecs = eigsh(lap, k=k, which='SM')
    return torch.tensor(eigvecs, dtype=torch.float)

def add_laplacian_pe(pyg_data, graph, k=10):
    """
    Add Laplacian Positional Encoding (PE) to the PyG Data object.
    """
    laplacian_pe = compute_laplacian_pe(graph, k)
    pyg_data.x = torch.cat([pyg_data.x, laplacian_pe], dim=1)
    return pyg_data


def add_positional_and_structural_encodings(pyg_data, graph, rw_steps=3, k=10):
    """
    Adds positional and structural encodings to a PyG graph data object.
    """
    
    pyg_data = extend_data_w_rwse(pyg_data, graph, steps=rw_steps)

    # Step 2: Add Node Degree Features
    pyg_data = add_node_degree_features(pyg_data, graph)

    # Step 3: Add Laplacian Positional Encodings (PE)
    pyg_data = add_laplacian_pe(pyg_data, graph, k=k)

    return pyg_data