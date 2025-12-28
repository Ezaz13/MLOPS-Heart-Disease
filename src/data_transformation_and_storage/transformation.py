import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
from datetime import datetime

from src.data_transformation_and_storage.sql import DatabaseConfig, create_connection, SQLQueries

# ------------------ SETUP ------------------
# Make paths robust by defining them relative to the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# Setup logging
logger = setup_logging("data_transformation_and_storage")

# Define path for the input CSV file
prepared_csv_file = os.path.join(PROJECT_ROOT, "data", "prepared", "prepared_churn_data_latest.csv")

def perform_feature_engineering(df):
    """
    Performs feature engineering by creating new, insightful features.
    This function now includes additional derived and aggregated features.
    """
    logger.info("Starting feature engineering...")
    df_transformed = df.copy()

    # Drop customerID as it is not a feature for machine learning models
    if 'customerID' in df_transformed.columns:
        df_transformed.drop('customerID', axis=1, inplace=True)
        logger.info("Dropped customerID column.")

    # --- Notes on Requested Features ---
    # - 'Total spend per customer' is represented by the 'TotalCharges' column.
    # - 'Customer tenure' is represented by the 'tenure' column.
    # - 'Activity frequency' cannot be derived as the dataset is a snapshot and lacks time-series activity data.

    # --- Existing & Enhanced Feature Creation ---

    # 1. Ratio of Monthly to Total Charges
    if 'MonthlyCharges' in df_transformed.columns and 'TotalCharges' in df_transformed.columns:
        # Create a temporary series to avoid division by zero or by very small numbers in scaled data
        total_charges = df_transformed['TotalCharges'].replace(0, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            df_transformed['charge_ratio'] = (df_transformed['MonthlyCharges'] / total_charges).round(4)
        
        # Replace infinite values (from division by near-zero) and NaNs with the column median
        df_transformed['charge_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_transformed['charge_ratio'].fillna(df_transformed['charge_ratio'].median(), inplace=True)
        logger.info("Created feature: charge_ratio")

    # 2. Number of Optional Services (Aggregation)
    optional_services = [
        'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 
        'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes'
    ]
    existing_optional_services = [col for col in optional_services if col in df_transformed.columns]
    if existing_optional_services:
        df_transformed['num_optional_services'] = df_transformed[existing_optional_services].sum(axis=1).astype(int)
        logger.info("Created feature: num_optional_services (aggregated optional services)")

    # --- New Derived Features ---

    # 3. Tenure-Charge Interaction
    # This feature can help models capture combined effects of tenure and monthly charges.
    if 'tenure' in df_transformed.columns and 'MonthlyCharges' in df_transformed.columns:
        df_transformed['tenure_charge_interaction'] = df_transformed['tenure'] * df_transformed['MonthlyCharges']
        logger.info("Created feature: tenure_charge_interaction")

    # 4. Tenure Groups
    # Binning tenure into categories can help capture non-linear relationships.
    # Since the data is scaled, we use quantiles for binning.
    if 'tenure' in df_transformed.columns:
        try:
            df_transformed['tenure_group'] = pd.qcut(
                df_transformed['tenure'], 
                q=4, 
                labels=['tenure_q1', 'tenure_q2', 'tenure_q3', 'tenure_q4'], 
                duplicates='drop'
            )
            # One-hot encode the new categorical feature
            df_transformed = pd.get_dummies(df_transformed, columns=['tenure_group'], prefix='', prefix_sep='')
            logger.info("Created features from tenure groups.")
        except Exception as e:
            logger.warning(f"Could not create tenure groups. Error: {e}")

    # 5. High-Value Customer
    # A flag for customers who might be considered high-value based on tenure and charges (top 25th percentile).
    if 'tenure' in df_transformed.columns and 'MonthlyCharges' in df_transformed.columns:
        high_tenure_threshold = df_transformed['tenure'].quantile(0.75)
        high_charge_threshold = df_transformed['MonthlyCharges'].quantile(0.75)
        df_transformed['is_high_value'] = (
            (df_transformed['tenure'] > high_tenure_threshold) & 
            (df_transformed['MonthlyCharges'] > high_charge_threshold)
        ).astype(int)
        logger.info("Created feature: is_high_value")

    logger.info("Feature engineering completed.")
    return df_transformed

def store_data_in_sqlite(df):
    """
    Stores the transformed DataFrame in a SQLite database using centralized config.
    """
    logger.info(f"Storing transformed data in SQLite database at {DatabaseConfig.DB_FILE}...")
    conn = None
    try:
        conn = create_connection()
        logger.info(f"Successfully connected to SQLite database: {DatabaseConfig.DB_FILE}")
        
        table_name = DatabaseConfig.TABLE_NAME
        
        # Explicitly drop the table to ensure a fresh start
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        logger.info(f"Table '{table_name}' dropped if it existed.")
        
        # Write the new data, let it fail if table exists (which it shouldn't)
        df.to_sql(table_name, conn, if_exists='fail', index=False)
        logger.info(f"Data successfully written to table '{table_name}'.")

        conn.commit()
        logger.info("Database transaction committed.")

        # Verify that the columns in the database match the DataFrame
        logger.info("Verifying table schema...")
        cursor.execute(f"PRAGMA table_info({table_name})")
        db_columns = [row[1] for row in cursor.fetchall()]
        
        df_columns = df.columns.tolist()
        
        if set(df_columns) == set(db_columns):
            logger.info("Schema verification successful: DataFrame columns match database columns.")
        else:
            logger.warning("Schema mismatch detected!")
            logger.warning(f"DataFrame columns: {sorted(df_columns)}")
            logger.warning(f"Database columns:  {sorted(db_columns)}")

        generate_schema_script(conn)
        print_deliverables()

    except Error as e:
        logger.error(f"Error connecting to or writing to SQLite database: {e}")
        raise CustomException(f"SQLite error: {e}", sys)
    finally:
        if conn:
            conn.close()
            logger.info("SQLite connection closed.")

def generate_schema_script(conn):
    """
    Generates a .sql file with the CREATE TABLE statement using centralized config.
    """
    logger.info(f"Generating SQL schema script at {DatabaseConfig.SCHEMA_FILE}...")
    try:
        with open(DatabaseConfig.SCHEMA_FILE, 'w') as f:
            f.write(f"-- SQL Schema for table: {DatabaseConfig.TABLE_NAME}\n")
            f.write(f"-- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            cursor = conn.cursor()
            cursor.execute(SQLQueries.GET_TABLE_SCHEMA, (DatabaseConfig.TABLE_NAME,))
            create_table_statement = cursor.fetchone()[0]
            f.write(create_table_statement + ";\n")
        logger.info(f"SQL schema script successfully saved to {DatabaseConfig.SCHEMA_FILE}")
        print(f"\n✅ SQL schema script saved to: {DatabaseConfig.SCHEMA_FILE}")
    except Exception as e:
        logger.error(f"Could not generate schema script: {e}")
        raise CustomException(f"Could not generate schema script: {e}", sys)

def print_deliverables():
    """
    Prints sample queries and a summary of the transformation logic using centralized queries.
    """
    print("\n--- DATABASE DELIVERABLES ---")
    
    # 1. Sample Queries
    print("\n1. Sample Queries to Retrieve Transformed Data:")
    print("\n  a) Retrieve the first 5 customers who have churned:")
    print(f"     {SQLQueries.SampleQueries.GET_CHURNED_CUSTOMERS}")
    
    print("\n  b) Get the average number of optional services for churned vs. non-churned customers:")
    print(f"     {SQLQueries.SampleQueries.AVG_SERVICES_BY_CHURN}")

    print("\n  c) Retrieve customers with a high charge ratio (monthly / total):")
    print(f"     {SQLQueries.SampleQueries.HIGH_CHARGE_RATIO}")

    print("\n  d) Retrieve high-value customers who have not churned:")
    print(f"     SELECT customerID, tenure, MonthlyCharges FROM {DatabaseConfig.TABLE_NAME} WHERE is_high_value = 1 AND Churn = 0 LIMIT 5;")

    # 2. Summary of Transformation Logic
    print("\n2. Summary of Transformation Logic Applied:")
    print(f"  - Loaded the dataset from '{os.path.basename(prepared_csv_file)}'.")
    print("  - Created 'charge_ratio': The ratio of MonthlyCharges to TotalCharges.")
    print("  - Created 'num_optional_services': Sum of all optional services subscribed to by a customer (aggregation).")
    print("  - Derived 'tenure_charge_interaction': Product of tenure and MonthlyCharges to capture interaction effects.")
    print("  - Derived 'tenure_group' features: Binned tenure into quantile-based groups and one-hot encoded them.")
    print("  - Derived 'is_high_value' flag: For customers with high tenure and high monthly charges (top 25th percentile).")
    print(f"  - Stored the final transformed data in the '{DatabaseConfig.TABLE_NAME}' table in '{os.path.basename(DatabaseConfig.DB_FILE)}'.")
    print(f"  - Saved the database schema to '{os.path.basename(DatabaseConfig.SCHEMA_FILE)}'.")

def main():
    """
    Main function to run the data transformation and storage pipeline.
    """
    logger.info("Starting Data Transformation and Storage Pipeline...")

    try:
        logger.info(f"Loading prepared data from {prepared_csv_file}...")
        if not os.path.exists(prepared_csv_file):
            raise FileNotFoundError(f"The prepared data file was not found at {prepared_csv_file}. Please run the data preparation script first.")
        df_prepared = pd.read_csv(prepared_csv_file)
        logger.info("Prepared data loaded successfully.")

        df_transformed = perform_feature_engineering(df_prepared)
        store_data_in_sqlite(df_transformed)

        logger.info("Data Transformation and Storage Pipeline Completed Successfully.")
        print("\nDATA TRANSFORMATION AND STORAGE PIPELINE COMPLETED SUCCESSFULLY")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    except CustomException as e:
        logger.error(f"A pipeline error occurred: {e}")
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
