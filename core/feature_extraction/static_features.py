import os
import pefile
import hashlib
import yara
import pandas as pd
import logging

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


def compile_yara_rules(directory):
    """
    Compile YARA rules from a directory.

    Parameters:
        directory (str): Path to the directory containing .yara or .yar files.

    Returns:
        yara.Rules: Compiled YARA rules.
    """
    rule_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
             if file.endswith('.yar') or file.endswith('.yara'):
                rule_path = os.path.join(root, file)
                namespace = os.path.splitext(file)[0]
                try:
                    # Test-compiling the individual rule to ensure it's valid
                    yara.compile(filepath=rule_path)
                    rule_files[namespace] = rule_path
                except yara.SyntaxError as e:
                    logging.error(f"Skipping invalid rule: {rule_path}. Error: {e}")

    if not rule_files:
        raise FileNotFoundError("No YARA rule files found in the specified directory.")

    return yara.compile(filepaths=rule_files)


def calculate_sha256(file_path):
    """
    Calculate SHA256 hash of a file.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        str: SHA256 hash of the file.
    """
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def calculate_section_entropy(pe):
    """
    Calculate entropy for each section in the PE file.

    Parameters:
        pe (pefile.PE): Loaded PE file.

    Returns:
        list[float]: List of entropy values for each section.
    """
    return [section.get_entropy() for section in pe.sections]


def is_packed(section_entropies):
    """
    Determine if a file is packed based on section entropies.

    Parameters:
        section_entropies (list[float]): Entropy values for sections.

    Returns:
        bool: True if the file is packed, False otherwise.
    """
    high_entropy_sections = [entropy for entropy in section_entropies if entropy > 7.0]
    return len(high_entropy_sections) > 0


def analyze_with_yara(file_path, yara_rules):
    """
    Analyze the sample with YARA.

    Parameters:
        file_path (str): Path to the file to analyze.
        yara_rules (yara.Rules): Precompiled YARA rules.

    Returns:
        dict: Dictionary of YARA matches.
    """
    if not yara_rules:
        return {"yara_matches": []}

    try:
        matches = yara_rules.match(file_path)
        return {"yara_matches": [match.rule for match in matches]} if matches else {"yara_matches": []}
    except yara.Error as e:
        logging.error(f"YARA analysis failed for {file_path}: {e}")
        return {"yara_matches": []}


def flatten_and_encode_yara(data, column):
    """
    Flatten and one-hot encode YARA rule lists in a DataFrame column.

    Parameters:
        data (pd.DataFrame): DataFrame containing a column with lists of YARA matches.
        column (str): The name of the column to process.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded YARA features.
    """
    # Ensure the column exists
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Explode the column and drop NaN or empty values
    exploded = data[column].explode()
    exploded = exploded.dropna()

    # Check if there are any values to encode
    if exploded.empty:
        logging.error(f"No non-empty values found in column '{column}'. Returning original DataFrame.")
        return data.drop(columns=[column])

    # One-hot encode the flattened values
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(exploded.values.reshape(-1, 1))

    # Create a DataFrame with encoded features
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))

    # Aggregate the encoded features back to the original DataFrame structure
    encoded_df = encoded_df.groupby(exploded.index).sum()

    # Drop the original column and merge the encoded features
    data = data.drop(columns=[column]).reset_index(drop=True)
    data = pd.concat([data, encoded_df.reset_index(drop=True)], axis=1)

    return data


def extract_static_features(data_path, yara_rules_dir=None):
    """
    Extract static features from malware samples.

    Parameters:
        data_path (str): Path to the malware samples directory.
        yara_rules_dir (str): Path to the directory containing YARA rules (optional).

    Returns:
        pd.DataFrame: DataFrame containing static features and YARA analysis results.
    """
    # Compile YARA rules if provided
    yara_rules = None
    if yara_rules_dir:
        if not os.path.exists(yara_rules_dir):
            raise FileNotFoundError(f"YARA rules directory not found: {yara_rules_dir}")
        yara_rules = compile_yara_rules(yara_rules_dir)

    # Initialize list to store features
    results = []

    # Process all files in the directory
    for file_name in tqdm(os.listdir(data_path), desc="Processing samples", unit="file"):
        file_path = os.path.join(data_path, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            # Extract static features
            pe = pefile.PE(file_path)
            section_entropies = calculate_section_entropy(pe)
            static_features = {
                "file_name": file_name,
                "sha256": calculate_sha256(file_path),
                "file_size": os.path.getsize(file_path),
                "num_sections": len(pe.sections),
                "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
                "image_base": pe.OPTIONAL_HEADER.ImageBase,
                "is_packed": is_packed(section_entropies),
                "avg_section_entropy": sum(section_entropies) / len(section_entropies) if section_entropies else 0.0,
            }

            # Add YARA matches
            yara_results = analyze_with_yara(file_path, yara_rules)
            static_features.update(yara_results)

            results.append(static_features)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    # Convert results to DataFrame
    features_df = pd.DataFrame(results)

    # Flatten and encode YARA matches if they exist
    if "yara_matches" in features_df.columns:
        features_df = flatten_and_encode_yara(features_df, "yara_matches")

    return features_df
