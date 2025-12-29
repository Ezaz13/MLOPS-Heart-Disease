"""
Feature Retrieval for the Heart Disease Prediction Model.

This module provides functions to retrieve engineered features from the SQLite
database, which acts as our feature store. It allows for easy access to
the transformed data for model_building training and inference.
"""

import os
import sys
import pandas as pd
import sqlite3
from sqlite3 import Error

# Make paths robust by defining them relative to the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


from src.data_transformation_and_storage.transformation import DatabaseConfig
from src.utility.logger import setup_logging

# Setup logging
logger = setup_logging("feature_retrieval")

def get_features_for_training(feature_list=None):
    """
    Retrieves the full dataset or a specified list of features from the feature store.

    Args:
        feature_list (list, optional): A list of feature names to retrieve.
                                      If None, all features are retrieved.

    Returns:
        pandas.DataFrame: A DataFrame containing the requested features.
    """
    logger.info(f"Connecting to the feature store at {DatabaseConfig.DB_FILE}...")
    conn = None
    try:
        conn = sqlite3.connect(DatabaseConfig.DB_FILE)
        logger.info("Successfully connected to the feature store.")

        table_name = DatabaseConfig.TABLE_NAME

        if feature_list:
            # Ensure no malicious feature names are passed
            safe_features = [f for f in feature_list if f.replace("_", "").isalnum()]
            query = f"SELECT {', '.join(safe_features)} FROM {table_name}"
            logger.info(f"Executing query: {query}")
        else:
            query = f"SELECT * FROM {table_name}"
            logger.info(f"Executing query to retrieve all features.")

        df = pd.read_sql_query(query, conn)
        logger.info(f"Successfully retrieved {len(df)} records with {len(df.columns)} columns.")
        return df

    except Error as e:
        logger.error(f"Error connecting to or reading from the feature store: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Feature store connection closed.")


if __name__ == '__main__':
    """
    Example of how to use the feature retrieval functions.
    This demonstrates a sample API for feature retrieval.
    """
    print("--- FEATURE STORE RETRIEVAL ---")

    # --- 1. Retrieve all features ---
    print("\n1. Retrieving all available features for model_building training...")
    try:
        all_features_df = get_features_for_training()
        print("Successfully retrieved all features.")
        print("Shape of the DataFrame:", all_features_df.shape)
        print("First 5 rows:")
        print(all_features_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")

    # --- 2. Retrieve a specific subset of features ---
    print("\n2. Retrieving a specific list of features for inference...")
    try:
        specific_features = ["age", "thalach", "rate_pressure_product", "is_high_risk", "target"]
        print(f"Requesting features: {specific_features}")
        
        specific_features_df = get_features_for_training(feature_list=specific_features)
        print("Successfully retrieved specific features.")
        print("Shape of the DataFrame:", specific_features_df.shape)
        print("First 5 rows:")
        print(specific_features_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")

    # --- 3. Documentation and Metadata ---
    print("\n3. Accessing Feature Metadata...")
    try:
        from feature_definitions import get_feature_definitions
        
        all_definitions = get_feature_definitions()
        print(f"Total number of defined features: {len(all_definitions)}")
        
        print("\nSample feature definition for 'rate_pressure_product':")
        rpp_def = next((f for f in all_definitions if f['name'] == 'rate_pressure_product'), None)
        if rpp_def:
            for key, value in rpp_def.items():
                print(f"  - {key}: {value}")
    except Exception as e:
        print(f"An error occurred while accessing metadata: {e}")
