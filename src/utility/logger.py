import logging
import os
from datetime import datetime


def setup_logging(log_prefix="app"):
    """
    Sets up logging to save log files in a 'logs' directory within the project root.
    The project root is determined automatically as the parent directory of 'src'.

    Args:
        log_prefix (str): The prefix to use for the log file name.
    """
    # Determine project root automatically, assuming this file is in a 'src' directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create the full path for the logs directory inside the project root
    logs_path = os.path.join(project_root, 'logs')
    os.makedirs(logs_path, exist_ok=True)

    # Define the log file name and its full path
    log_file_name = f"{log_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(logs_path, log_file_name)

    # Get a specific logger instance
    logger = logging.getLogger(log_prefix)
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers if the function is called multiple times
    if not logger.handlers:
        # Create a file handler to write to the log file
        handler = logging.FileHandler(log_file_path)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

    return logger
