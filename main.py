import argparse
import os
import yaml

from workflows.malware_clustering_analysis import MalwareClusteringWorkflow

def load_config(config_path):
    """
    Load the YAML configuration file.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def main():
    parser = argparse.ArgumentParser(
        description="Malware Research Framework: Analyze malware data and generate insights."
    )
    parser.add_argument(
        "--workflow",
        choices=["malware_clustering_analysis" ],
        required=True,
        help="Specify the workflow to execute: 'malware_clustering_analysis'."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base_config.yaml",
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    config = load_config(args.config)

    print(f"Starting workflow: {args.workflow}")
    print(f"Using configuration file: {args.config}")

    if args.workflow == "malware_clustering_analysis":
        MalwareClusteringWorkflow(config).run()
    # elif args.workflow == "actor_mapping":
    #    run_actor_mapping(args.config)
    # elif args.workflow == "anomaly_detection":
    #    run_anomaly_detection(args.config)
    else:
        print("Invalid workflow specified. Use --help for options.")
        return

    print(f"Workflow {args.workflow} completed successfully!")

if __name__ == "__main__":
    if not os.path.exists("config/base_config.yaml"):
        print("Configuration file not found. Ensure 'config/base_config.yaml' exists.")
        exit(1)

    main()
