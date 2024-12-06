import pandas as pd
import os
import logging

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
        Normalize and optionally weight features for clustering in malware analysis.

        This step is important for malware analysis because:
        1. **Normalization:** Malware datasets often contain features with varying scales (e.g., byte counts, opcode frequencies, string lengths).
        Without normalization, features with larger numerical ranges might dominate the clustering process, leading to biased results.
        Normalization ensures fair contribution of all features during clustering by preventing large-scale features from overshadowing smaller-scale ones.
        2. **Feature weighting:** Some features may have higher relevance in distinguishing malware families or threat actors.
        Applying weights allows emphasizing these important features based on domain knowledge or experimental findings.

        Args:
            feature_data (DataFrame): The raw features extracted from the malware samples.

        Returns:
            DataFrame: Preprocessed features with normalized values and optional weights applied.
        """
        numeric_features = feature_data.select_dtypes(include=["number"])
        logging.debug("Normalizing features...")

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numeric_features)
        normalized_df = pd.DataFrame(normalized_data, columns=numeric_features.columns)

        weights = self.config["feature_extraction"].get("feature_weights", None)
        if weights:
            logging.debug("Applying feature weights...")
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

        dynamic_features_enabled = config["feature_extraction"]["dynamic_features"]["enabled"]
        # dynamic_features_file = config["feature_extraction"]["dynamic_features"]["output_file"]

        feature_dfs = []

        if static_features_enabled:
            if os.path.exists(static_features_file):
                logging.debug(f"Loading static features from {static_features_file}...")
                static_features = pd.read_csv(static_features_file)
            else:
                yara_rules_dir = config["feature_extraction"]["static_features"]["yara_rules_dir"]
                logging.debug("Extracting static features...")
                static_features = extract_static_features(data_path, yara_rules_dir)
                static_features.to_csv(static_features_file, index=False)
                logging.info(f"Static features saved to {static_features_file}.")
            feature_dfs.append(static_features)
        if dynamic_features_enabled:
            logging.error("Dynamic features extraction not implemented yet.")

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
        cluster_means = data.groupby(cluster_labels).mean()
        top_features_by_variance = cluster_means.var().sort_values(ascending=False).head(10)
        return cluster_composition, top_features_by_variance

    def evaluate_clustering(self, data, cluster_labels):
        """
        Evaluate clustering quality using the silhouette score.
        """
        score = silhouette_score(data, cluster_labels)
        logging.info(f"Silhouette Score: {score:.2f} (Higher is better, range: -1 to 1)")

    def save_results(self, data, cluster_labels, file_suffix="clustering_results"):
        """
        Save clustering results to a CSV file.
        """
        data["Cluster"] = cluster_labels
        output_file = os.path.join(self.output_path, f"{file_suffix}.csv")
        data.to_csv(output_file, index=False)
        return output_file

    def run(self):
        """
        Placeholder method for running the workflow. To be implemented by derived classes.
        """
        raise NotImplementedError("The `run` method must be implemented in the derived class.")
