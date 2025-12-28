import os
import sys
import pandas as pd
import numpy as np
import warnings
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ------------------ WARNING SUPPRESSION ------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ SETUP ------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# Setup logging
logger = setup_logging("data_preparation")

# Define data paths
raw_data_path = os.path.join(PROJECT_ROOT, "data", "raw", "uci")
prepared_data_path = os.path.join(PROJECT_ROOT, "data", "prepared")
eda_folder = os.path.join(PROJECT_ROOT, "artifacts", "eda")

# Create necessary directories
os.makedirs(prepared_data_path, exist_ok=True)
os.makedirs(eda_folder, exist_ok=True)

def get_latest_csv_file(folder_path):
    """
    Get the latest CSV file from a folder based on modification time.
    """
    list_of_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not list_of_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return max(list_of_files, key=os.path.getctime)


def load_dataset():
    """
    Load the Heart Disease dataset from the raw data folder.
    """
    try:
        logger.info("Loading dataset from local raw data folder...")

        # Load dataset
        # Assuming the raw data is in the raw_data_path
        csv_file = get_latest_csv_file(raw_data_path)
        logger.info(f"Loading dataset: {csv_file}")
        df = pd.read_csv(csv_file)
        
        logger.info(f"Size of dataset: {df.shape}")
        print(f"Size of dataset: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise CustomException(f"Error loading dataset: {e}", sys)

def clean_data(df):
    """
    Performs initial data cleaning for the Heart Disease dataset.
    Handles missing values ('?') and normalizes the target variable.
    """
    logger.info("Performing initial data cleaning...")
    df_clean = df.copy()

    # Replace '?' with NaN which is common in UCI datasets
    df_clean.replace('?', np.nan, inplace=True)

    # Ensure numeric columns are actually numeric
    # Common numeric columns in Heart Disease dataset that might be read as object due to '?'
    numeric_candidates = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
    for col in numeric_candidates:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Normalize Target Variable
    # UCI dataset often uses 'num' (0-4) or 'target'. We want binary classification (0 vs 1).
    target_col = 'target' if 'target' in df_clean.columns else 'num'
    if target_col in df_clean.columns:
        df_clean['target'] = df_clean[target_col].apply(lambda x: 1 if x > 0 else 0)
        if target_col != 'target':
            df_clean.drop(columns=[target_col], inplace=True)
        logger.info("Normalized target variable to binary (0: No Disease, 1: Disease).")
        
    return df_clean

def perform_eda(df, eda_path):
    """
    Perform enhanced Exploratory Data Analysis using Seaborn and save artifacts.
    """
    logger.info("Starting Enhanced Exploratory Data Analysis (EDA)...")
    
    # Set Seaborn style for all plots
    sns.set_theme(style="whitegrid")

    # Summary Statistics
    summary_stats = df.describe(include='all')
    summary_stats.to_csv(os.path.join(eda_path, "summary_statistics.csv"))
    logger.info(f"Saved summary statistics to {eda_path}")

    # Identify column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = 'target'

    # Remove identifiers and target from feature lists
    if target_col in numeric_cols: numeric_cols.remove(target_col)

    # --- VISUALIZATIONS ---

    # 1. Distribution of Numeric Features vs. Target
    logger.info("Generating distribution plots for numeric features by Target...")
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue=target_col, kde=True, multiple="stack")
        plt.title(f'Distribution of {col} by Target', fontsize=14)
        plt.savefig(os.path.join(eda_path, f'{col}_distribution_by_target.png'))
        plt.close()

    # 2. Box Plots of Numeric Features vs. Target for outlier detection
    logger.info("Generating box plots for numeric features by Target...")
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f'Box Plot of {col} by Target', fontsize=14)
        plt.savefig(os.path.join(eda_path, f'{col}_boxplot_by_target.png'))
        plt.close()

    # 3. Distribution of Categorical Features vs. Target
    logger.info("Generating count plots for categorical features by Target...")
    for col in categorical_cols:
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=col, hue=target_col)
        plt.title(f'Distribution of {col} by Target', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(eda_path, f'{col}_distribution_by_target.png'))
        plt.close()

    # 4. Correlation Matrix of Numeric Features and Target
    logger.info("Generating correlation heatmap for numeric features...")
    corr_cols = numeric_cols + [target_col]
    # Impute missing values for correlation matrix calculation
    corr_df = df[corr_cols].copy()
    for col in corr_df.columns:
        if corr_df[col].isnull().any():
            corr_df[col].fillna(corr_df[col].median(), inplace=True)
    
    corr_matrix = corr_df.corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features and Target', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_path, 'numeric_correlation_heatmap.png'))
    plt.close()

    logger.info(f"Saved enhanced EDA visualizations to {eda_path}")

def preprocess_and_save_data(df, output_path):
    """
    Handle missing values, scale numeric features, one-hot encode categorical features,
    and save the final dataset.
    """
    logger.info("Starting final preprocessing...")
    df_processed = df.copy()

    # Define feature types for Heart Disease dataset
    # Note: 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal' are categorical/ordinal
    # 'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca' are numeric
    
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    target_col = 'target'

    # Ensure categorical columns are present in the dataframe
    categorical_features = [col for col in categorical_features if col in df_processed.columns]
    numeric_features = [col for col in numeric_features if col in df_processed.columns]

    # Impute missing values
    if numeric_features:
        numeric_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_features] = numeric_imputer.fit_transform(df_processed[numeric_features])
        logger.info("Imputed missing values in numeric features with median.")

    if categorical_features:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_features] = categorical_imputer.fit_transform(df_processed[categorical_features])
        logger.info("Imputed missing values in categorical columns with mode.")

    # Standardize numerical attributes
    if numeric_features:
        scaler = StandardScaler()
        df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
        logger.info("Standardized numerical attributes.")

    # One-hot encode categorical variables, ensuring output is 0/1
    if categorical_features:
        # Convert to string/object first to ensure get_dummies treats them as categorical
        # (since some might be integers like cp: 1,2,3,4)
        for col in categorical_features:
            df_processed[col] = df_processed[col].astype(str)
            
        df_processed = pd.get_dummies(df_processed, columns=categorical_features, drop_first=False, dtype=int)
        logger.info("One-hot encoded categorical variables with 0/1 output.")

    # Final check on the dataset size
    logger.info(f"Size of the final merged and preprocessed dataset: {df_processed.shape}")
    print(f"Size of the final merged and preprocessed dataset: {df_processed.shape}")
    
    # Save the prepared dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"prepared_heart_data_{timestamp}.csv"
    output_file_path = os.path.join(output_path, file_name)
    df_processed.to_csv(output_file_path, index=False)
    
    logger.info(f"Prepared dataset saved to: {output_file_path}")
    print(f"\n Prepared dataset saved to: {output_file_path}")
    
    # For easy access, also save a 'latest' version
    latest_path = os.path.join(output_path, "prepared_heart_data_latest.csv")
    df_processed.to_csv(latest_path, index=False)
    logger.info(f"Latest prepared dataset copy saved to: {latest_path}")


def main():
    """
    Main function to run the data preparation pipeline.
    """
    logger.info("Starting Data Preparation Pipeline...")

    # Step 1: Load dataset
    df_raw = load_dataset()

    # Step 2: Initial data cleaning
    df_cleaned = clean_data(df_raw)

    # Step 3: Perform EDA on cleaned data
    eda_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eda_run_folder = os.path.join(eda_folder, f"eda_run_{eda_timestamp}")
    os.makedirs(eda_run_folder, exist_ok=True)
    perform_eda(df_cleaned, eda_run_folder)

    # Step 4: Preprocess data and save the final version
    preprocess_and_save_data(df_cleaned, prepared_data_path)

    logger.info("Data Preparation Pipeline Completed Successfully.")
    print("\nDATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == '__main__':
    try:
        main()
    except CustomException as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
