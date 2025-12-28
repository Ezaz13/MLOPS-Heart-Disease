"""
Data Versioning with DVC.

This script automates the process of versioning datasets using DVC (Data Version Control).
It provides functions to initialize DVC, track new data files, and commit changes to DVC.

To run this script, ensure you have DVC installed:
  pip install dvc
"""

import os
import subprocess
import sys

# Make paths robust by defining them relative to the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utility.logger import setup_logging

# Setup logging
logger = setup_logging("data_versioning")

def find_latest_file(directory, prefix):
    """Finds the most recently modified file in a directory with a given prefix."""
    logger.info(f"Searching for latest file with prefix '{prefix}' in '{directory}'...")
    try:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix)]
        if not files:
            logger.warning(f"No files found with prefix '{prefix}' in '{directory}'.")
            return None
        latest_file = max(files, key=os.path.getmtime)
        logger.info(f"Found latest file: {latest_file}")
        return latest_file
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return None

# --- Define Data Files to Version ---
# We will version the raw datasets, the prepared dataset, and the final transformed database.
DATA_FILES_TO_VERSION = [
    # Raw data files are now added dynamically in the main function.
    os.path.join(PROJECT_ROOT, "data", "prepared", "prepared_churn_data_latest.csv"),
    os.path.join(PROJECT_ROOT, "data", "database", "customer_churn.db")
]

def run_command(command, cwd=PROJECT_ROOT):
    """Executes a shell command and logs its output."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        return True
    except FileNotFoundError:
        logger.error(f"Command '{command[0]}' not found. Please ensure DVC is installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def initialize_dvc():
    """Initializes DVC in the project root if it hasn't been already."""
    dvc_dir = os.path.join(PROJECT_ROOT, ".dvc")
    if os.path.exists(dvc_dir):
        logger.info("DVC has already been initialized.")
        return True
    
    logger.info("Initializing DVC repository...")
    # --no-scm prevents DVC from looking for a Git repository.
    # If you use Git, you can remove this flag.
    return run_command(["dvc", "init", "--no-scm"])

def version_data_file(file_path):
    """Adds a data file to DVC tracking."""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at: {file_path}")
        return False

    logger.info(f"Tracking file with DVC: {os.path.relpath(file_path, PROJECT_ROOT)}")
    # Use relpath for cleaner DVC file names
    return run_command(["dvc", "add", os.path.relpath(file_path, PROJECT_ROOT)])

def main():
    """
    Main function to run the data versioning pipeline.
    """
    logger.info("Starting Data Versioning Pipeline...")

    # 1. Initialize DVC
    if not initialize_dvc():
        print("\nError: DVC initialization failed. Please check the logs.")
        sys.exit(1)

    # 2. Dynamically find the latest raw files and add them to the versioning list
    kaggle_dir = os.path.join(PROJECT_ROOT, "data", "raw", "kaggle")
    huggingface_dir = os.path.join(PROJECT_ROOT, "data", "raw", "huggingface")

    latest_kaggle_file = find_latest_file(kaggle_dir, "customer_churn_dataset_kaggle")
    latest_huggingface_file = find_latest_file(huggingface_dir, "customer_churn_dataset_huggingface")

    if latest_kaggle_file:
        DATA_FILES_TO_VERSION.insert(0, latest_kaggle_file)
    if latest_huggingface_file:
        DATA_FILES_TO_VERSION.insert(0, latest_huggingface_file)

    # 3. Version the data files
    success_count = 0
    for file_path in DATA_FILES_TO_VERSION:
        if version_data_file(file_path):
            success_count += 1
        else:
            print(f"\nError: Failed to version {file_path}. Please check the logs.")

    if success_count < len(DATA_FILES_TO_VERSION):
        print("\nData versioning pipeline completed with errors.")
        sys.exit(1)

    # 4. Show DVC status and explain deliverables
    print("\n--- DVC STATUS ---")
    run_command(["dvc", "status"])
    
    print("\n--- DVC DELIVERABLES ---")
    print("DVC has created/updated .dvc files for your tracked data.")
    print("You should now commit these files to your version control system (e.g., Git).")
    print("The generated .dvc files are located alongside the data files they track.")

    logger.info("Data Versioning Pipeline Completed Successfully.")
    print("\nDATA VERSIONING PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == '__main__':
    main()
