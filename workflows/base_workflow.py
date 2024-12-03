import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from core.clustering.kmeans import kmeans_clustering
from core.clustering.hdbscan import hdbscan_clustering
from core.feature_extraction.static_features import extract_static_features
# from core.feature_extraction.dynamic_features import extract_dynamic_features

class BaseWorkflow:
    """
    Base class for workflows with common functionality like feature extraction, preprocessing, clustering, and analysis.
    """
    def __init__(self, config):
        self.config = config
        self.output_path = config["general"]["output_path"]

    def preprocess_features(self, feature_data):
        """
        Normalize and optionally weight features for clustering.
        """
        numeric_features = feature_data.select_dtypes(include=["number"])
        print("Normalizing features...")
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numeric_features)
        normalized_df = pd.DataFrame(normalized_data, columns=numeric_features.columns)

        weights = self.config["feature_extraction"].get("feature_weights", None)
        if weights:
            print("Applying feature weights...")
            for feature, weight in weights.items():
                if feature in normalized_df.columns:
                    normalized_df[feature] *= weight

        return normalized_df

    def extract_features(self, config, data_path):
        """
        Extract features based on the configuration.
        """
        static_features_enabled = config["feature_extraction"]["static_features"]["enabled"]
        static_features_file = config["feature_extraction"]["static_features"]["output_file"]

        feature_dfs = []

        if static_features_enabled:
            if os.path.exists(static_features_file):
                print(f"Loading static features from {static_features_file}...")
                static_features = pd.read_csv(static_features_file)
            else:
                yara_rules_dir = config["feature_extraction"]["static_features"]["yara_rules_dir"]
                print("Extracting static features...")
                static_features = extract_static_features(data_path, yara_rules_dir)
                static_features.to_csv(static_features_file, index=False)
                print(f"Static features saved to {static_features_file}.")
            feature_dfs.append(static_features)

        if feature_dfs:
            return pd.concat(feature_dfs, axis=1)
        else:
            raise ValueError("No features extracted. Check the data and configuration.")

    def perform_clustering(self, preprocessed_data):
        """
        Perform clustering using the specified algorithm.
        """
        algorithm = self.config["clustering"]["algorithm"]
        if algorithm == "kmeans":
            return kmeans_clustering(preprocessed_data, self.config["clustering"]["num_clusters"], self.config["clustering"]["random_state"])
        elif algorithm == "hdbscan":
            return hdbscan_clustering(preprocessed_data, self.config["clustering"].get("min_cluster_size", 5))
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    def analyze_clusters(self, data, cluster_labels):
        """
        Analyze clusters to provide insights.
        """
        cluster_composition = data.groupby(cluster_labels).size()
        print("Cluster Composition:")
        print(cluster_composition)

        cluster_means = data.groupby(cluster_labels).mean()
        print("Top Features by Variance Across Clusters:")
        print(cluster_means.var().sort_values(ascending=False).head(10))

    def evaluate_clustering(self, data, cluster_labels):
        """
        Evaluate clustering quality using the silhouette score.
        """
        score = silhouette_score(data, cluster_labels)
        print(f"Silhouette Score: {score:.2f}")

    def save_results(self, data, cluster_labels, file_suffix="clustering_results"):
        """
        Save clustering results to a CSV file.
        """
        data["Cluster"] = cluster_labels
        output_file = os.path.join(self.output_path, f"{file_suffix}.csv")
        data.to_csv(output_file, index=False)
        print(f"Clustering results saved to {output_file}")

    def run(self):
        """
        Placeholder method for running the workflow. To be implemented by derived classes.
        """
        raise NotImplementedError("The `run` method must be implemented in the derived class.")
