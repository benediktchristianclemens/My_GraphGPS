import scanpy as sc # type: ignore
import pandas as pd  # type: ignore
import os 
# Below the actual function out-commented. For Comparison of the clustering I needed exactly 12 clusters
# def create_sc_data(sct_file="data_pbmc/pca_reduced_data.csv", n_neighbors=15, resolution=1.0):
#     """
#     Reads SCT-transformed data from a CSV file, performs PCA, and uses Louvain clustering to assign clusters.

#     Args:
#         sct_file (str): Path to the SCT-transformed data CSV file.
#         n_neighbors (int): Number of neighbors for the KNN graph.
#         resolution (float): Resolution for Louvain clustering.

#     Returns:
#         AnnData: Annotated data matrix with PCA and Louvain clustering results.
#     """
#     # Load SCT-transformed data
#     sct_data = pd.read_csv(sct_file, index_col=0)

#     # Convert to AnnData
#     adata = sc.AnnData(X=sct_data.values, obs=pd.DataFrame(index=sct_data.index), var=pd.DataFrame(index=sct_data.columns))

#     # PCA
#     sc.tl.pca(adata, svd_solver='arpack')

#     # Create KNN graph and perform Louvain clustering
#     sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=10)
#     sc.tl.louvain(adata, resolution=resolution)

#     return adata, sct_data

def create_sc_data(sct_file="data_pbmc/pca_reduced_data.csv", n_neighbors=15, target_clusters=12):
    """
    Reads SCT-transformed data from a CSV file, performs PCA, and uses Louvain clustering to assign a target number of clusters.

    Args:
        sct_file (str): Path to the SCT-transformed data CSV file.
        n_neighbors (int): Number of neighbors for the KNN graph.
        target_clusters (int): Desired number of clusters.

    Returns:
        AnnData: Annotated data matrix with PCA and Louvain clustering results.
    """

    base_dir = os.path.dirname(__file__)  # Directory of import_pbmc.py
    sct_file_path = os.path.join(base_dir, sct_file)

    # Check if the file exists
    if not os.path.exists(sct_file_path):
        raise FileNotFoundError(f"File not found: {sct_file_path}")

    # Load the data
    sct_data = pd.read_csv(sct_file_path, index_col=0)


    # Convert to AnnData
    adata = sc.AnnData(
        X=sct_data.values,
        obs=pd.DataFrame(index=sct_data.index),
        var=pd.DataFrame(index=sct_data.columns)
    )

    # PCA
    sc.tl.pca(adata, svd_solver="arpack")

    # Create KNN graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=10)

    # Optimize resolution to achieve target_clusters
    resolution = 0.5  # Start with a baseline resolution
    max_iterations = 20  # Limit iterations to avoid excessive tuning
    for _ in range(max_iterations):
        sc.tl.louvain(adata, resolution=resolution)
        num_clusters = len(adata.obs["louvain"].unique())
        if num_clusters == target_clusters:
            break
        elif num_clusters < target_clusters:
            resolution += 0.1
        else:
            resolution -= 0.1

    print(f"Optimized resolution: {resolution:.2f} with {num_clusters} clusters.")
    return adata, sct_data