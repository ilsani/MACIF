import os
import matplotlib.pyplot as plt
import numpy as np
import logging

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text


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
    if hasattr(data, "select_dtypes"):
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        if numeric_data.empty:
            raise ValueError("No numeric data available for clustering. Check the features.")
    else:
        numeric_data = data  # Assume it's already a NumPy array
    
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
    # cumulative_variance = explained_variance.cumsum()
    #logging.info(f"Explained variance by components: {explained_variance}")
    #logging.info(f"Cumulative explained variance: {cumulative_variance}")

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=50,
        alpha=0.8,
        edgecolor='black'
    )
    
    # Create a 3D scatter plot
    #fig = plt.figure(figsize=(12, 8))
    #ax = fig.add_subplot(111, projection='3d')
    #scatter = ax.scatter(
    #    reduced_data[:, 0],
    #    reduced_data[:, 1],
    #    reduced_data[:, 2],
    #    c=cluster_labels,
    #    cmap="viridis",
    #    s=50,
    #    alpha=0.8,
    #    edgecolor='black'
    #)

    #colorbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title("PCA Visualization of Clusters")
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}% variance)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}% variance)")
    


    # Annotate points with cluster labels (optional)
    # for i, txt in enumerate(cluster_labels):
    #    plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8, alpha=0.7)

     # Annotate centroids
    # for i, (x, y) in enumerate(centroids):
        # plt.annotate(f"Cluster {i}", (x, y), fontsize=10, color="red", fontweight="bold")
    #     plt.annotate(
     #        f"Cluster {i}",
      #       (x, y),
       #      fontsize=10,
   #          color="red",
    #         fontweight="bold",
    #         bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white")
    #     )
    
    # Annotate centroids with dynamic adjustment to avoid overlap
    texts = []
    for i, (x, y) in enumerate(centroids):
        texts.append(
            plt.text(
                x, y,
                f"{i}",
                fontsize=10,
                color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white")
            )
        )
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    #for i, (x, y, z) in enumerate(centroids):
    #    ax.text(
    #        x, y, z,
    #        f"Cluster {i}",
    #        fontsize=10,
    #        color="red",
    #        fontweight="bold",
    #        bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white")
    #    )

    #ax.set_title("PCA Visualization of Clusters (3D)")
    #ax.set_xlabel(f"Principal Component 1 ({explained_variance[0] * 100:.2f}% variance)")
    #ax.set_ylabel(f"Principal Component 2 ({explained_variance[1] * 100:.2f}% variance)")
    #ax.set_zlabel(f"Principal Component 3 ({explained_variance[2] * 100:.2f}% variance)")


    # Save plot
    os.makedirs(output_path, exist_ok=True)
    plot_path = os.path.join(output_path, "pca_clusters.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"PCA cluster visualization saved to: {plot_path}")
