from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_clustering(data, num_clusters, random_state=42, n_init=10):
    """
    Perform KMeans clustering on the given data and calculate the silhouette score.

    Parameters:
        data (DataFrame or ndarray): The input data for clustering.
        num_clusters (int): The number of clusters to create.
        random_state (int): Seed for reproducibility.

    Returns:
        model (KMeans): Trained KMeans model.
        cluster_labels (ndarray): Cluster labels for the input data.
        silhouette (float): Silhouette score to evaluate clustering quality.
    """
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    if numeric_data.empty:
        raise ValueError("No numeric data available for clustering. Check the features.")
    
    # Train KMeans model
    model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=n_init)
    cluster_labels = model.fit_predict(numeric_data)

    return model, cluster_labels