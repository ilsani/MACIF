import pandas as pd
import numpy as np
import os
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from core.visualization.cluster_visualization import visualize_clusters_pca


class ClusterManager:
    """
    A class to encapsulate all clustering-related logic for feature evaluation, 
    clustering, outlier detection, and relationship identification.
    """

    def __init__(self, config):
        self.config = config
        self.clusters = None
        self.cluster_labels = None
        self.silhouette_score = None
        self.cluster_composition = None
        self.top_feature_importance_by_variance = None
        self.top_feature_importance_by_rf = None
        self.top_feature_importance_overlap = None
        self.outliers = None

    def perform_clustering(self, data):
        """
        Perform clustering using the specified algorithm in the configuration.

        Parameters:
            data (pd.DataFrame): Preprocessed data to cluster.

        Returns:
            tuple: The clustering model and cluster labels.
        """
        algorithm = self.config["clustering"]["algorithm"]
        if algorithm == "kmeans":
            from core.clustering.kmeans import kmeans_clustering
            self.clusters, self.cluster_labels = kmeans_clustering(
                data,
                num_clusters=self.config["clustering"]["num_clusters"],
                random_state=self.config["clustering"]["random_state"]
            )
        elif algorithm == "hdbscan":
            from core.clustering.hdbscan import hdbscan_clustering
            self.clusters, self.cluster_labels = hdbscan_clustering(
                data,
                min_cluster_size=self.config["clustering"].get("min_cluster_size", 5)
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        return self

    def evaluate_clustering(self, data):
        """
        Evaluate clustering quality using the silhouette score.

        Parameters:
            data (pd.DataFrame): Preprocessed data.
            cluster_labels (array-like): Cluster labels assigned to the data.

        Returns:
            float: Silhouette score.
        """
        self.silhouette_score = silhouette_score(data, self.cluster_labels)
        return self

    def evaluate_feature_importance(self, data):
        """
        Evaluate feature importance globally using two methods:
        1. Random Forest Classifier-based importance.
        2. Variance-based importance per cluster.

        Parameters:
            data (pd.DataFrame): Input data.
            cluster_labels (array-like): Cluster labels for the data.

        Returns:
            tuple: Cluster composition, top features by both methods.
        """
        self.cluster_composition = data.groupby(self.cluster_labels).size()

        # Method 1: Random Forest Feature Importance
        rf = RandomForestClassifier(random_state=42)
        rf.fit(data, self.cluster_labels)
        self.top_feature_importance_by_rf = pd.Series(rf.feature_importances_, index=data.columns).sort_values(ascending=False).head(10)

        # Method 2: Variance-Based Feature Importance
        cluster_means = data.groupby(self.cluster_labels).mean()
        self.top_feature_importance_by_variance = cluster_means.var().sort_values(ascending=False).head(10)

        self.compare_feature_importance()

        return self

    def compare_feature_importance(self):
        """
        Compare feature importance rankings from two methods.

        Parameters:
            tfbv1 (pd.Series): Top features by Random Forest.
            tfbv2 (pd.Series): Top features by variance.

        Returns:
            tuple: Spearman rank correlation, p-value, and overlap of top features.
        """
        self.top_feature_importance_overlap = set(self.top_feature_importance_by_rf.index).intersection(set(self.top_feature_importance_by_variance.index))
        self.rank_corr, self.p_value = spearmanr(self.top_feature_importance_by_rf, self.top_feature_importance_by_variance)
        return self

    def detect_outliers(self, features):
        """
        Detect outliers based on distances to cluster centroids.

        Parameters:
            data (pd.DataFrame): Preprocessed data.
            cluster_labels (array-like): Cluster labels for the data.
            model: The clustering model.

        Returns:
            pd.DataFrame: Outlier detection results with distances.
        """
        distances = np.linalg.norm(features.values - self.clusters.cluster_centers_[self.cluster_labels], axis=1)
    
        # Define outlier threshold as the 95th percentile of distances
        threshold = np.percentile(distances, 95)
        
        # Identify outliers
        outlier_flags = distances > threshold
        
        # Save only the outliers in self.outliers
        self.outliers = pd.DataFrame({
            "distance_to_centroid": distances[outlier_flags],
            "data_index": np.arange(len(features))[outlier_flags],
            "cluster_id": self.cluster_labels[outlier_flags],  # Include cluster IDs for outliers
        }).set_index("data_index")

        return self
    
    def describe_outliers(self):
        """
        Describe the outlier detection results in English.

        Returns:
            str: A summary of the outlier detection results in English.
        """
        # Ensure there are outliers to describe
        if self.outliers.empty:
            return "No outliers were detected in the dataset."

        # Total number of outliers
        total_outliers = len(self.outliers)

        # Maximum and mean distance of outliers to their centroids
        max_distance = self.outliers['distance_to_centroid'].max()
        mean_distance = self.outliers['distance_to_centroid'].mean()

        # Group outliers by cluster ID
        outliers_by_cluster = self.outliers.groupby('cluster_id').size()

        # Generate the summary
        summary = (
            f"Outlier Detection Summary:\n"
            f"- Total number of outliers detected: {total_outliers}\n"
            f"- Maximum distance to centroid among outliers: {max_distance:.4f}\n"
            f"- Average distance to centroid among outliers: {mean_distance:.4f}\n"
            f"- Outliers detected across clusters:\n"
        )

        for cluster_id, count in outliers_by_cluster.items():
            summary += f"  - Cluster {cluster_id}: {count} outliers\n"

        # List outlier indices and their cluster IDs
        outlier_details = self.outliers.reset_index()[['data_index', 'cluster_id', 'distance_to_centroid']]
        summary += "\nOutlier Details:\n" + outlier_details.to_string(index=False)

        return summary

    def identify_relationships(self, similarity_threshold=0.7):
        """
        Identify relationships between clusters based on similarity.

        Parameters:
            data (pd.DataFrame): Preprocessed data.
            cluster_labels (array-like): Cluster labels for the data.
            similarity_threshold (float): Threshold to consider clusters related.

        Returns:
            tuple: Relationship matrix and similarity threshold.
        """
        cluster_means = self.clusters.groupby(self.cluster_labels).mean()
        similarity_matrix = cosine_similarity(cluster_means)

        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=cluster_means.index,
            columns=cluster_means.index
        )

        relationships = (similarity_df >= similarity_threshold).astype(int)
        return relationships, similarity_threshold

    def get_feature_importance_comparison_description(self):
        summary = []
        summary.append("Comparison of Feature Importance Evaluation:")
        summary.append("1. Top features v. 1:\n{}".format(self.top_feature_importance_by_rf))
        summary.append("1. Top features v. 2:\n{}".format(self.top_feature_importance_by_variance))
        summary.append(f"3. Overlap in top 10 features: {len(self.top_feature_importance_overlap)} out of 10")
       
        if self.rank_corr is not None:
            summary.append(f"4. Spearman rank correlation between rankings: {self.rank_corr:.2f} (p-value: {self.p_value:.3f})")

        if len(self.top_feature_importance_overlap) > 5:
            summary.append("The overlap in top features suggests that both methods largely agree.")
        else:
            summary.append("The limited overlap suggests that the two methods capture different aspects of the data.")

        if self.rank_corr > 0.7:
            summary.append("The high rank correlation indicates strong agreement between the two methods.")
        elif 0.3 < self.rank_corr <= 0.7:
            summary.append("The moderate rank correlation suggests partial agreement between the methods.")
        else:
            summary.append("The low rank correlation suggests that the methods are identifying different patterns.")

        return "\n".join(summary)
    
    def save_results(self, features, output_path, file_suffix="clustering_results"):
        """
        Save clustering results to a CSV file.
        """
        features["Cluster"] = self.cluster_labels
        output_file = os.path.join(output_path, f"{file_suffix}.csv")
        features.to_csv(output_file, index=False)
        return output_file
    
    def visualize_clusters_pca(self, features, output_path, file_suffix="clustering_results"):
        output_file = os.path.join(output_path, f"{file_suffix}")
        visualize_clusters_pca(features, self.cluster_labels, output_file)
        return output_file