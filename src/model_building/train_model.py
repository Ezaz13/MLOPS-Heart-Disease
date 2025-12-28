"""
Model Training and Evaluation for Customer Churn Prediction using MLflow.

This script trains and evaluates machine learning models to predict customer churn.
It uses MLflow to track experiments, log metrics, and version models with robust signatures.

1.  Loads features from the feature store.
2.  Prepares the data for training (splitting, encoding, type casting).
3.  Trains multiple algorithms (Logistic Regression, Random Forest).
4.  Logs parameters, metrics, and models (with signatures) to MLflow for each run.
5.  Generates a model performance report.
"""

import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Make paths robust by defining them relative to the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to the Python path
sys.path.insert(0, PROJECT_ROOT)

from src.feature_store.feature_retrieval import get_features_for_training
from src.utility.logger import setup_logging

# Setup logging
logger = setup_logging("model_building")

# --- Configuration ---
# Define the path for the performance report
REPORT_FILE = os.path.join(PROJECT_ROOT, "src", "model_building", "model_performance_report.md")
MLFLOW_EXPERIMENT_NAME = "Customer Churn Prediction"

TARGET_VARIABLE = "Churn"

def prepare_data(df):
    """Prepares the data for model training."""
    logger.info("Preparing data for training...")

    # Drop rows with missing target variable
    df.dropna(subset=[TARGET_VARIABLE], inplace=True)

    # Encode the target variable
    if df[TARGET_VARIABLE].dtype == 'object':
        le = LabelEncoder()
        df[TARGET_VARIABLE] = le.fit_transform(df[TARGET_VARIABLE])
        logger.info(f"Target variable '{TARGET_VARIABLE}' encoded.")

    # Separate features (X) and target (y)
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]

    # Ensure all feature columns are numeric
    X = X.select_dtypes(include=['number'])

    # Cast all integer features to float64 to avoid potential schema errors at inference
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col]):
            X[col] = X[col].astype('float64')
    logger.info("Casted integer feature columns to float64 to prevent future schema errors.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Data split into training and testing sets.")
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains, evaluates, and logs models using MLflow."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    performance_results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            logger.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log parameters to MLflow
            mlflow.log_params(model.get_params())
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Create an input example to infer the model signature
            input_example = X_train.head()
            
            # Log the model to MLflow with an input example to infer the signature
            mlflow.sklearn.log_model(
                sk_model=model,
                name=name,
                input_example=input_example
            )
            logger.info(f"Model {name} logged to MLflow with signature.")

            # Store results for the report
            performance_results[name] = {
                "run_id": run.info.run_id,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_report": classification_report(y_test, y_pred)
            }
            logger.info(f"{name} evaluation complete.")

    return performance_results

def generate_performance_report(results):
    """Generates a markdown report of the model performance."""
    with open(REPORT_FILE, 'w') as f:
        f.write("# Model Performance Report\n\n")
        f.write(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"*MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'\n\n")
        
        for name, result in results.items():
            f.write(f"## {name}\n\n")
            f.write(f"- **MLflow Run ID**: `{result['run_id']}`\n")
            f.write(f"- **Accuracy**: {result['accuracy']:.4f}\n")
            f.write(f"- **Precision**: {result['precision']:.4f}\n")
            f.write(f"- **Recall**: {result['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {result['f1_score']:.4f}\n\n")
            f.write("### Classification Report\n\n")
            f.write("```\n")
            f.write(result['classification_report'])
            f.write("\n```\n\n")
    
    logger.info(f"Performance report saved to: {REPORT_FILE}")

def main():
    """
    Main function to run the model training and evaluation pipeline.
    """
    logger.info("Starting Model Building Pipeline with MLflow...")

    try:
        # --- FIX for Windows Pathing Issue ---
        # Manually construct the file URI for the MLflow tracking location
        mlruns_path = os.path.join(PROJECT_ROOT, "mlruns")
        # Replace backslashes with forward slashes for URI compatibility
        tracking_uri = "file:///" + mlruns_path.replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Set MLflow tracking URI to: {mlflow.get_tracking_uri()}")

        # Set MLflow experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"Using MLflow experiment: '{MLFLOW_EXPERIMENT_NAME}'")

        # 1. Load features
        features_df = get_features_for_training()
        
        # 2. Prepare data
        X_train, X_test, y_train, y_test = prepare_data(features_df)
        
        # 3. Train, evaluate, and log models
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # 4. Generate performance report
        generate_performance_report(results)

        print("\n--- MODEL BUILDING DELIVERABLES (with MLflow) ---")
        print(f"1. Python script for training: src/model_building/train_model.py")
        print(f"2. Model performance report: {os.path.relpath(REPORT_FILE, PROJECT_ROOT)}")
        print(f"3. Versioned models saved in MLflow.")
        print("\nTo view your versioned models, run `mlflow ui` in your terminal and navigate to the experiment.")

        logger.info("Model Building Pipeline Completed Successfully.")
        print("\nMODEL BUILDING PIPELINE COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error(f"An error occurred during the model building pipeline: {e}", exc_info=True)
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
