# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from my_graphgps.data.import_pbmc import create_sc_data
from my_graphgps.data.create_cell_to_cell_graph import create_pyg_data_from_adata
from my_graphgps.network.encodings import (
    extend_data_w_rwse,
    add_node_degree_features,
    add_laplacian_pe,
    add_positional_and_structural_encodings
)
from my_graphgps.training.hyperparam_tuning import hyperparameter_tuning
from my_graphgps.training.test_best_model import train_and_evaluate_best_model
from my_graphgps.training.cluster_embeddings import visualize_cluster_embeddings
import logging
from tqdm import tqdm

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data into train, validation, and test masks.

    Parameters:
    ----------
    data : torch_geometric.data.Data
        The PyG Data object.
    train_ratio : float, optional (default=0.7)
        Proportion of data to use for training.
    val_ratio : float, optional (default=0.15)
        Proportion of data to use for validation.
    test_ratio : float, optional (default=0.15)
        Proportion of data to use for testing.
    random_seed : int, optional (default=42)
        Seed for random number generator for reproducibility.

    Returns:
    -------
    data : torch_geometric.data.Data
        The PyG Data object with added train_mask, val_mask, and test_mask.
    """
    np.random.seed(random_seed)
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    test_size = num_nodes - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    return data

def main():
    # Step 1: Import PBMC data
    logging.info("Step 1: Importing PBMC data...")
    adata, sct_data = create_sc_data()
    logging.info(f"Data imported with shape: {adata.shape}")

    # Step 2: Create PyG Data object from AnnData
    logging.info("Step 2: Creating PyG Data object...")
    data, graph = create_pyg_data_from_adata(adata)
    logging.info(f"PyG Data object created with {data.num_nodes} nodes and {data.num_features} features.")

    # Step 3: Add encodings to the data
    logging.info("Step 3: Adding encodings to the data...")
    # Create different data variants with various encodings
    data_rwse = extend_data_w_rwse(data, graph)  # Structural encodings
    data_and_encodings = add_positional_and_structural_encodings(data, graph)  # All encodings
    logging.info("Encodings added. Data variants created.")

    # Step 4: Split data into train, validation, and test masks
    logging.info("Step 4: Splitting data into train, validation, and test sets...")
    data = split_data(data)
    data_rwse = split_data(data_rwse)  # Structural encodings only
    data_and_encodings = split_data(data_and_encodings)  # All encodings

    logging.info("Data splitting completed.")
    all_data = [
        ("data", data),
        ("data_rwse", data_rwse),
        ("data_and_encodings", data_and_encodings)
    ]

    for name, dataset in all_data:
        logging.info(f"Dataset '{name}':")
        logging.info(f"Train nodes: {dataset.train_mask.sum().item()}, Validation nodes: {dataset.val_mask.sum().item()}, Test nodes: {dataset.test_mask.sum().item()}")

    # Step 5: Define hyperparameter grid
    param_grid = {
        "hidden_channels": [32, 64, 128],
        "heads": [2, 4, 8],
        "dropout": [0.2, 0.5],
        "learning_rate": [0.001, 0.01, 0.1],
        "epochs": [50, 150, 200]
    }
    logging.info("Hyperparameter grid defined.")

    # Iterate over each data variant in all_data
    for name, dataset in tqdm(all_data, desc='Processing Datasets', unit='dataset'):
        logging.info(f"=== Processing Dataset: {name} ===")

        # Step 6: Determine number of features and classes
        num_features = dataset.num_features
        num_classes = int(dataset.y.max().item()) + 1  # assuming labels start at 0
        logging.info(f"Number of features: {num_features}")
        logging.info(f"Number of classes: {num_classes}")

        # Step 7: Perform hyperparameter tuning
        logging.info("Step 7: Starting hyperparameter tuning...")
        best_params, best_acc = hyperparameter_tuning(
            curr_data=dataset,
            num_features=num_features,
            num_classes=num_classes,
            param_grid=param_grid,
            num_samples=30,
            random_seed=42
        )
        logging.info(f"Best Parameters for '{name}': {best_params}")
        logging.info(f"Best Test Accuracy for '{name}': {best_acc:.4f}")
        best_hidden_channels, best_heads, best_dropout, best_learning_rate, best_epochs = best_params
        # Step 8: Train and evaluate the best model
        logging.info("Step 8: Training the best model with optimal hyperparameters...")
        best_model = train_and_evaluate_best_model(
            curr_data=dataset,
            num_features=num_features,
            num_classes=num_classes,
            best_params=best_params,
            data_name=name  # Pass the dataset name for plotting
        )
        logging.info("Best model training completed.")

        # Step 9: Perform clustering and visualize embeddings
        logging.info("Step 9: Performing clustering and visualizing embeddings...")
        visualize_cluster_embeddings(
            best_model=best_model,
            curr_data=dataset,
            n_clusters=12,
            data_name=name  # Pass the dataset name for plotting
        )
        logging.info("Clustering and visualization completed.")

    logging.info("All datasets processed successfully.")

if __name__ == "__main__":
    main()
