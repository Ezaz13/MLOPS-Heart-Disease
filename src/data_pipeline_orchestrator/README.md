# Data Pipeline Orchestrator with Prefect

This document describes the automated data pipeline for the customer churn prediction project, orchestrated using **Prefect**. This approach provides a robust, modern, and scalable way to define, schedule, and monitor the entire workflow.

## Pipeline Overview

The pipeline automates all the steps required to go from raw data to a trained and versioned machine learning model. It is designed as a **Prefect flow**, where each major step is a **Prefect task**. This creates a formal Directed Acyclic Graph (DAG) that clearly defines the dependencies between tasks.

### The Prefect DAG

The orchestrator executes the following sequence of tasks. Prefect ensures that each task (a `run_script` instance) only begins after its upstream dependencies have completed successfully.

1.  **Data Ingestion** (`run_script`)
    -   **Description**: Ingests raw data from all sources.

2.  **Data Validation** (`run_script`)
    -   **Depends On**: Data Ingestion
    -   **Description**: Validates the raw data against quality standards.

3.  **Data Preparation** (`run_script`)
    -   **Depends On**: Data Validation
    -   **Description**: Cleans and merges the validated data.

4.  **Data Transformation** (`run_script`)
    -   **Depends On**: Data Preparation
    -   **Description**: Applies feature engineering and saves data to the feature store.

5.  **Data Versioning** (`run_script`)
    -   **Depends On**: Data Transformation
    -   **Description**: Versions the datasets using DVC.

6.  **Model Building** (`run_script`)
    -   **Depends On**: Data Versioning
    -   **Description**: Trains models and logs them to MLflow.

## How to Run the Pipeline

To execute the entire automated pipeline, follow these steps:

1.  **Install Prefect**: If you haven't already, install Prefect from PyPI.

    ```bash
    pip install prefect
    ```

2.  **Open Your Terminal**: Open a terminal or command prompt.

3.  **Navigate to the Project Root**: It is crucial to run the orchestrator from the main project directory to ensure tools like DVC and MLflow work correctly.

    ```bash
    cd D:\Bits\Semester_2\dmml
    ```

4.  **Run the Pipeline Flow**: Execute the `pipeline.py` script.

    ```bash
    python src/data_pipeline_orchestrator/pipeline.py
    ```

### Monitoring and Failure Handling

When you run the script, Prefect will execute the flow. You will see real-time logs in your console from the `get_run_logger()`. Prefect automatically handles task failures and will stop the downstream tasks if a dependency fails. The `@task` decorator is also configured to retry a failed task once before marking it as permanently failed.

For a more advanced visual interface, you can use the Prefect UI. Run `prefect server start` in your terminal and navigate to the displayed URL to see a dashboard of your runs.

## Deliverables

*   **Pipeline DAG/Script**: The `pipeline.py` script is the main deliverable, showcasing task automation and dependency management using Prefect's `@task` and `@flow` decorators.
*   **Documentation**: This `README.md` file provides the necessary documentation for understanding and running the Prefect-based pipeline.
