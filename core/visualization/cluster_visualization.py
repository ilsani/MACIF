import os
import matplotlib.pyplot as plt
import numpy as np
import logging

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def visualize_clusters_pca(data, cluster_labels, output_path):
    """
    Perform PCA for dimensionality reduction and visualize clusters.

    Parameters:
        data (pd.DataFrame or np.ndarray): The numeric input data for PCA.
        cluster_labels (array-like): Cluster labels assigned to the data.
        output_path (str): Directory to save the visualization.

    Returns:
        None
    """

    logging.debug("Visualizing clusters PCA ...")

    # Filter numeric data
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    if numeric_data.empty:
        raise ValueError("No numeric data available for clustering. Check the features.")
    
    # Standardize the data
    logging.debug("Standardizing the data...")
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numeric_data)
    
    # Perform PCA
    logging.debug("Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(standardized_data)

    # Calculate centroids for clusters
    centroids = np.array([reduced_data[cluster_labels == i].mean(axis=0) for i in np.unique(cluster_labels)])

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    logging.debug(f"Explained variance by components: {explained_variance}")

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.7,
        edgecolor='k'
    )
    plt.colorbar(scatter, label="Cluster Label")
    plt.title("PCA Visualization of Clusters")
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}% variance)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}% variance)")


    # Annotate points with cluster labels (optional)
    # for i, txt in enumerate(cluster_labels):
    #    plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8, alpha=0.7)

     # Annotate centroids
    for i, (x, y) in enumerate(centroids):
        plt.annotate(f"Cluster {i}", (x, y), fontsize=10, color="red", fontweight="bold")



    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save plot
    plot_path = os.path.join(output_path, "pca_clusters.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"PCA cluster visualization saved to: {plot_path}")
