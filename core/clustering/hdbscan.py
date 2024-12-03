import hdbscan

def hdbscan_clustering(data, min_cluster_size=5):
    """
    Perform HDBSCAN clustering on the given data.

    Parameters:
        data (DataFrame or ndarray): The input data for clustering.
        min_cluster_size (int): Minimum size of clusters.

    Returns:
        model (HDBSCAN): Trained HDBSCAN model.
        cluster_labels (ndarray): Cluster labels for the input data (-1 for noise).
    """
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = model.fit_predict(data)
    return model, cluster_labels
