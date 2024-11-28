import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Optional
# from import_pbmc import create_sc_data

def create_pyg_data_from_adata(
    adata,
    alpha: float = 0.8,
    k: int = 10,
    max_iter: int = 50,
    tol: float = 1e-4,
    n_neighbors: int = 9,
    label_key: str = 'louvain'
) -> Data:
    """
    Convert an AnnData object to a PyTorch Geometric Data object with a graph
    based on enhanced cell similarity.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix containing observations (cells) and variables (genes).
    alpha : float, optional (default=0.8)
        Damping factor for network enhancement.
    k : int, optional (default=10)
        Number of iterations for network enhancement.
    max_iter : int, optional (default=50)
        Maximum number of iterations for the network enhancement algorithm.
    tol : float, optional (default=1e-4)
        Tolerance for convergence in the network enhancement algorithm.
    n_neighbors : int, optional (default=9)
        Number of nearest neighbors to connect in the graph.
    label_key : str, optional (default='louvain')
        The key in `adata.obs` to use for node labels.

    Returns
    -------
    data : torch_geometric.data.Data
        The PyG Data object containing node features, edge indices, edge weights,
        and labels suitable for graph-based machine learning.
    
    Raises
    ------
    ValueError
        If the specified label_key is not found in `adata.obs`.
    """
    
    # Step 1: Remove variables (genes) with zero variance
    zero_variance_mask = adata.X.var(axis=0) != 0  # Typically, variance is computed across cells (axis=0)
    if isinstance(adata.X, np.ndarray):
        adata = adata[:, zero_variance_mask]
    else:
        # If adata.X is a sparse matrix
        adata = adata[:, zero_variance_mask].copy()

    # Step 2: Compute cell similarity matrix using correlation coefficient
    # Transpose if cells are rows and genes are columns
    cell_similarity_matrix = np.corrcoef(adata.X)
    
    # Handle NaNs resulting from constant rows after zero-variance filtering
    cell_similarity_matrix = np.nan_to_num(cell_similarity_matrix)

    def network_enhancement(W: np.ndarray, alpha: float = 0.8, k: int = 10, 
                           max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
        """
        Perform network enhancement (Wang et al., 2018) on the similarity matrix.

        Parameters
        ----------
        W : np.ndarray
            The initial similarity matrix.
        alpha : float, optional (default=0.8)
            Damping factor.
        k : int, optional (default=10)
            Number of neighbors (not directly used in this function).
        max_iter : int, optional (default=50)
            Maximum number of iterations.
        tol : float, optional (default=1e-4)
            Tolerance for convergence.

        Returns
        -------
        P : np.ndarray
            The enhanced similarity matrix.
        """
        W = np.maximum(W, 0)  # Ensure non-negativity
        row_sums = W.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        W = W / row_sums  # Row-normalization

        P = W.copy()
        for iteration in range(max_iter):
            P_prev = P.copy()
            P = alpha * np.dot(P, W) + (1 - alpha) * W
            if np.linalg.norm(P - P_prev, ord='fro') < tol:
                print(f'Network enhancement converged after {iteration+1} iterations.')
                break
        else:
            print(f'Network enhancement reached maximum iterations ({max_iter}).')
        return P

    # Step 3: Enhance the similarity matrix
    enhanced_similarity_matrix = network_enhancement(
        W=cell_similarity_matrix, 
        alpha=alpha, 
        k=k, 
        max_iter=max_iter, 
        tol=tol
    )

    # Step 4: Construct the graph using NetworkX
    G = nx.Graph()
    n_cells = adata.shape[0]
    
    # Add nodes with features
    for i in range(n_cells):
        G.add_node(i, features=adata.X[i])

    # Add edges based on top k neighbors from the enhanced similarity matrix
    for i in range(n_cells):
        # Get indices of top n_neighbors + 1 (including self)
        neighbors = np.argsort(-enhanced_similarity_matrix[i])[:n_neighbors + 1]
        for neighbor in neighbors:
            if neighbor != i:
                weight = enhanced_similarity_matrix[i, neighbor]
                G.add_edge(i, neighbor, weight=weight)

    # Step 5: Convert NetworkX graph to PyG Data object
    # Create mapping from node to index
    node_to_index = {node: i for i, node in enumerate(G.nodes)}
    
    # Extract edge indices and weights
    edge_index = torch.tensor(
        [[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in G.edges],
        dtype=torch.long
    ).t().contiguous()
    
    edge_weight = torch.tensor(
        [G[edge[0]][edge[1]]['weight'] for edge in G.edges],
        dtype=torch.float
    )
    
    # Extract node features
    node_features = torch.tensor(
        np.array([G.nodes[node]['features'] for node in G.nodes]),
        dtype=torch.float
    )
    
    # Assign labels
    if label_key not in adata.obs:
        raise ValueError(f"Label key '{label_key}' not found in adata.obs.")
    
    y = torch.tensor(
        adata.obs[label_key].astype(int).values,
        dtype=torch.long
    )
    
    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, y=y)
    
    return data, G
