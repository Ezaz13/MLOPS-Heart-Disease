import os
import sys
import io
import time
import requests
from datetime import datetime

import pandas as pd

from src.utility.exception import CustomException
from src.utility.logger import setup_logging

# -------------------------------------------------------------------
# Setup logging
# -------------------------------------------------------------------
logger = setup_logging("ingestion")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

UCI_SOURCE_PATH = os.path.join(PROJECT_ROOT, "data/raw/uci")
UCI_FILE_NAME = "heart_disease_dataset"

# UCI Heart Disease (Cleveland) dataset
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column names as per UCI documentation
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

# -------------------------------------------------------------------
# Ingestion Logic
# -------------------------------------------------------------------
def ingest_from_uci(target_file_path: str):
    try:
        logger.info("Triggered data ingestion from UCI Machine Learning Repository...")
        logger.info(f"Downloading dataset from: {UCI_DATA_URL}")

        # UCI often blocks requests without a User-Agent header
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(UCI_DATA_URL, timeout=30, headers=headers)
        response.raise_for_status()

        # Convert response to DataFrame
        df = pd.read_csv(
            io.StringIO(response.text),
            header=None,
            names=UCI_COLUMNS
        )

        # Replace missing values marked as '?'
        df.replace("?", pd.NA, inplace=True)

        # Basic validation
        if df.empty:
            raise ValueError("Downloaded UCI dataset is empty.")

        # Save to CSV
        df.to_csv(target_file_path, index=False)
        logger.info(f"UCI Heart Disease dataset saved to '{target_file_path}'")

        # File size validation
        if os.path.getsize(target_file_path) < 1024:
            raise ValueError("Downloaded CSV file is too small.")

        logger.info("UCI data ingestion completed successfully.")
        return True

    except Exception as e:
        raise CustomException(e, sys)


def ingest_source_data(max_retries=3, delay=10):
    postfix_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_path = os.path.join(
        UCI_SOURCE_PATH,
        f"{UCI_FILE_NAME}_{postfix_datetime}.csv"
    )

    os.makedirs(UCI_SOURCE_PATH, exist_ok=True)

    attempt = 0
    while attempt < max_retries:
        try:
            logger.info("--- Starting Data Ingestion Cycle ---")
            ingest_from_uci(target_path)
            logger.info("--- Data Ingestion Cycle Finished Successfully ---")
            return True

        except Exception as e:
            attempt += 1
            logger.error(
                f"Ingestion attempt {attempt} failed: {str(e)}",
                exc_info=True
            )
            if attempt < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.critical("Data ingestion failed after maximum retries.")
                return False

    return False


# -------------------------------------------------------------------
# Standalone Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Running UCI Heart Disease ingestion script as standalone process.")
    success = ingest_source_data()

    if success:
        logger.info("Data ingestion completed successfully.")
        sys.exit(0)
    else:
        logger.error("Data ingestion failed.")
        sys.exit(1)
