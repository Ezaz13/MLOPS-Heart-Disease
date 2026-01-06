# MLOps Assignment Report
# Title: Heart Disease UCI Dataset

# Group 130

| Sl. No. | Name | BITS ID | Contribution |
| :--- | :--- | :--- | :--- |
| 1 | MD. EZAZUL HAQUE | 2024aa05083 | 100% |
| 2 | SWAPNIL BHUSHAN VERMA | 2024ab05216 | 100% |
| 3 | MAYANK SPARSH | 2024aa05386 | 100% |
| 4 | MD. SHAFI HUSSAIN | 2024ab05039 | 100% |
| 5 | M. MOHIT SHARMA | 2023ac05887 | 100% |


## 1. Project Introduction & Problem Statement

### 1.1 Background
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection and diagnosis are paramount for effective treatment and patient survival. Traditional diagnostic methods often rely on the subjective analysis of a doctor, which can be time-consuming and prone to human error. Machine Learning (ML) offers a data-driven approach to assist medical professionals by analyzing complex patterns in clinical data to predict the presence of heart disease.

### 1.2 Objective
The primary objective of this project is to build a robust, reproducible, and scalable **End-to-End MLOps Pipeline** for predicting heart disease. Unlike a simple notebook experiment, this project focuses on the engineering challenges of deploying ML in the real world:
-   **Automation**: reducing manual intervention in data processing and training.
-   **Reproducibility**: Ensuring every model version can be traced back to the exact code and data that generated it.
-   **Scalability**: Serving the model via a containerized API that can handle production loads.
-   **Reliability**: Implementing strict data validation and monitoring.

### 1.3 Dataset
We utilize the **UCI Heart Disease Dataset** (Cleveland database). This dataset serves as a benchmark in the ML community, containing patient attributes such as age, cholesterol levels, and exercise-induced angina, to predict the presence of heart disease (angiographic disease status).

---

## 2. Setup/Install Instructions

### 2.1 Prerequisites
To ensure a consistent development environment, the following tools are required:
-   **Operating System**: Windows 10/11, macOS, or Linux.
-   **Language**: Python 3.9 or higher.
-   **Version Control**: Git.
-   **Containerization**: Docker Desktop.

### 2.2 detailed Installation Guide
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Ezaz13/MLOPS-Heart-Disease.git
    cd Heart-Disease-Prediction-Project
    ```

2.  **Environment Setup**:
    It is critical to use a virtual environment to avoid dependency conflicts.
    ```bash
    python -m venv venv
    # Activate on Windows
    venv\Scripts\activate
    # Activate on Unix/macOS
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 2.3 Running the MLOps Pipeline
The entire lifecycle (Data Ingestion $\rightarrow$ Model Registration) is orchestrated via a single entry point. This script uses **Prefect** to manage task dependencies and retry logic.

```bash
python src/data_pipeline_orchestrator/pipeline.py
```

**Pipeline Execution Flow**:
1.  **Ingestion**: Downloads data from UCI.
2.  **Validation**: Checks data schemas with Great Expectations.
3.  **Preparation**: Cleans and imputes data.
4.  **Transformation**: Generates medical features.
5.  **Training**: Trains models and registers the best one in MLflow.

### 2.4 Serving the Model
To start the production-grade Flask API locally:
```bash
python src/model_serving/app.py
```

### 2.5 Validation & Testing
We provide tools to verify the deployment:
-   **Batch Test**: `python src/model_serving/validate_deployment.py` tests multiple patient profiles (High Risk, Low Risk, Edge Cases).
-   **Single Request**:
    ```bash
    curl -X POST http://localhost:5000/predict -d '{"age":63, "sex":1, "cp":3, "trestbps":145, "chol":233, "fbs":1, "restecg":0, "thalach":150, "exang":0, "oldpeak":2.3, "slope":0, "ca":0, "thal":1}'
    ```

---

## 3. Data Dictionary & Pipeline Details

### 3.1 Data Dictionary
Understanding the data is crucial for both modeling and medical interpretation.

| Feature | Description | Type | Key Values / Range |
| :--- | :--- | :--- | :--- |
| `age` | Patient's age in years | Numerical | 29 - 77 |
| `sex` | Biological Sex | Categorical | 0 = Female, 1 = Male |
| `cp` | Chest Pain Type | Categorical | 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal, 4 = Asymptomatic |
| `trestbps` | Resting Blood Pressure (mm Hg) | Numerical | 94 - 200 |
| `chol` | Serum Cholesterol (mg/dl) | Numerical | 126 - 564 |
| `fbs` | Fasting Blood Sugar > 120 mg/dl | Categorical | 0 = False, 1 = True (Diabetic indicator) |
| `restecg` | Resting ECG Results | Categorical | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy |
| `thalach` | Maximum Heart Rate Achieved | Numerical | 71 - 202 |
| `exang` | Exercise Induced Angina | Categorical | 0 = No, 1 = Yes |
| `oldpeak` | ST depression induced by exercise relative to rest | Numerical | 0.0 - 6.2 (Indicates ischemia) |
| `slope` | Slope of the peak exercise ST segment | Categorical | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
| `ca` | Number of major vessels colored by flourosopy | Numerical | 0 - 3 |
| `thal` | Thalassemia | Categorical | 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect |
| `target` | Diagnosis of Heart Disease | Target | 0 = No Disease, 1-4 = Disease Present |

### 3.2 Detailed Pipeline Stages

#### Stage 1: Robust Data Ingestion (`ingestion.py`)
-   **Source**: Downloads directly from the UCI repository archives.
-   **Reliability**: Implements a `retry` mechanism (3 attempts with backoff) to handle transient network failures.
-   **Versioning**: Files are saved with a timestamp (`heart_disease_dataset_YYYYMMDD_HHMMSS.csv`) to ensure we never overwrite raw data, enabling point-in-time recovery.

#### Stage 2: Quality Assurance with Great Expectations (`validation.py`)
We employ Great Expectations (GX) to enforce a strict **Data Contract** between the ingestion and modeling layers. This prevents "silent failures" where bad data flows downstream without raising an error.

-   **The Data Contract**:
    -   The pipeline expects exact adherence to the schema defined in `heart_disease_suite`.
    -   **Schema Integrity**: `expect_table_columns_to_match_ordered_list` ensures strictly ordered 14 columns.
-   **Statistical Guardrails & Logic**:
    -   **Age**: Validated to be medically realistic (`20` to `100`).
    -   **Blood Pressure (`trestbps`)**: Must be within survivable range `80-250`.
    -   **Categorical Integrity**: `sex` must be `{0, 1}`, `cp` must be `{1, 2, 3, 4}`.
-   **Automated Documentation (Data Docs)**:
    -   Every validation run generates a human-readable HTML report within `reports/`.
    -   This provides a persistent **Audit Trail**, allowing stakeholders to see exactly *why* a pipeline run failed (e.g., "Age column contained value '150'").
-   **Action**: If validation fails, the pipeline halts immediately (`sys.exit(1)`), preventing downstream corruption.

#### Stage 3: Data Preparation & Cleaning (`preparation.py`)
-   **Handling Missing Values**: The UCI dataset uses `?` for missing values. These are converted to `NaN`.
-   **Imputation Strategy**:
    -   **Median** for numerical columns (robust to outliers like high cholesterol).
    -   **Mode** for categorical columns (most frequent category).
-   **Target Normalization**: The raw target is 0-4. We binarize this to **0 (No Disease)** and **1 (Disease)** to frame the problem as a binary classification task.
-   **Output**: Cleaned data allows for accurate Exploratory Data Analysis (EDA).

#### Stage 4: Feature Engineering (`transformation.py`)
To improve model performance, we engineered domain-specific medical features:
1.  **Rate Pressure Product (RPP)**: $RPP = HeartRate \times SystolicBP$.
    -   *Logic*: A standard index of myocardial oxygen consumption and stress on the heart.
2.  **Metabolic Indicator**: `chol * fbs`.
    -   *Logic*: Captures the compounding risk of high cholesterol in diabetic patients.
3.  **High Risk Flag**: A composite binary flag.
    -   *Logic*: Identifies patients with both significant ST depression (`oldpeak` > 75th percentile) and blocked vessels (`ca` > 75th percentile), representing advanced arterial blockage.

---

## 4. Modeling Methodology & Experiment Tracking

### 4.1 Algorithm Selection
We trained and evaluated three distinct classes of algorithms to find the best fit:
1.  **Logistic Regression**: Provides a strong linear baseline and is highly interpretable (odds ratios).
2.  **Random Forest**: An ensemble method that handles non-linear relationships and feature interactions well without heavy tuning.
3.  **Gradient Boosting**: Often achieves state-of-the-art performance by sequentially correcting errors.

### 4.2 Training Configuration (`train_model.py`)
All models were trained using **Stratified K-Fold Cross-Validation (K=5)** to ensure class distribution remains consistent across folds.
-   **Data Balancing**: We used `class_weight='balanced'` for Logistic Regression and Random Forest to penalize misclassifying the minority class.
-   **Scaling**: A pipeline utilizing `StandardScaler` was used to normalize features, critical for Logistic Regression and Gradient Descent convergence.

### 4.3 Evaluation Metric: F1-Score
We prioritized the **F1-Score** over Accuracy.
-   **Medical Rationale**: In disease prediction, **False Negatives** (missed diagnosis) are dangerous. **False Positives** (unnecessary tests) are costly.
-   The F1-Score balances Precision and Recall. Our goal was to maximize Recall (catch all sick patients) while maintaining acceptable Precision.

### 4.4 Experiment Results (MLflow)
We utilized MLflow's tracking capabilities to log rich metadata for every run, ensuring that "Experimentation" is not "Chaos".

#### 4.4.1 Tracked Metadata
-   **Tags**: we applied specific tags to organize the experiments:
    -   `domain="healthcare"`: Allows filtering by business domain.
    -   `problem_type="binary_classification"`: Helps downstream tools understand the model output format.
-   **Parameters**: Complete hyperparameters configuration (e.g., `class_weight='balanced'`, `n_estimators`, `learning_rate`).
-   **Metrics**: We logged a holistic view of performance: `cv_roc_auc`, `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`.

#### 4.4.2 Comparative Results Table
| Model | Run ID | CV ROC-AUC | F1 Score | Recall | Config |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | `...e31d...` | 0.8840 | **0.8814** | **0.9286** | `n_estimators=100`, `max_depth=10` |
| Logistic Regression | `...1a69...` | **0.9009** | 0.8667 | 0.9286 | `C=0.1`, `solver='liblinear'` |
| Gradient Boosting | `...d0f4...` | 0.8528 | 0.8667 | 0.9286 | `lr=0.05`, `n_estimators=150` |

### 4.5 MLflow Lifecycle & Model Registry

The transition from a "Raw Experiment" to a "Production Model" is automated via the **Model Registry**.

1.  **Artifact Preservation**:
    For every run, we store:
    -   `model.pkl`: The serialized Scikit-learn pipeline (Preprocessor + Classifier).
    -   `conda.yaml`: The exact environment definition (dependencies) required to run the model.
    -   `input_example.json`: A sample of 5 rows from the training data. This acts as a schema contract for the serving API.

2.  **Champion Selection Logic**:
    The generic `train_model.py` script removes human bias from the selection process:
    -   It iterates through all training runs.
    -   It compares the **F1-Score** of each model.
    -   The model with the highest F1-Score is identified as the "Champion".

3.  **Registration**:
    The Champion model is programmatically registered to the MLflow Model Registry under the name `HeartDiseaseModel`. This provides a stable URI (`models:/HeartDiseaseModel/Production`) for the deployment services to consume, decoupling training from serving.

---

## 5. MLOps Infrastructure & Deployment

### 5.1 CI/CD Pipeline (GitHub Actions)
We implemented a "Quality Gate" philosophy in `.github/workflows/ci_cd_pipeline.yml`. No code reaches production without passing these gates:
1.  **Strict Linting (`flake8`)**:
    -   Enforces a max line length of 127.
    -   Blocks build on syntax errors (`E9`, `F63`, `F7`, `F82`).
    -   *Philosophy*: Code must be readable and error-free before testing begins.
2.  **Unit Tests (`pytest`)**:
    -   Discovers and runs all tests in the `tests/` directory.
    -   Ensures individual functions (like data cleaning logic) work as expected.
3.  **Integration Pipeline**:
    -   Executes the orchestrator (`pipeline.py`).
    -   Validates that the `ingestion -> validation -> training` flow is unbroken.
4.  **Artifact Archival (Audit Trail)**:
    -   **`mlflow-runs`**: Preserves the exact experiment logs.
    -   **`validation-reports`**: Saves Great Expectations output proof.
    -   **Policy**: Artifacts are uploaded `if: always()`, meaning even failed runs generates logs for debugging.

### 5.2 Containerization Strategy
Our `Dockerfile` is optimized for security and size:
-   **Base Image**: `python:3.10-slim` (Selected for minimal attack surface vs full Alpine compatibility issues).
-   **Optimizations**:
    -   `PYTHONDONTWRITEBYTECODE=1`: Prevents `.pyc` files, keeping the container clean.
    -   `PYTHONUNBUFFERED=1`: Ensures logs are flushed immediately to stdout for real-time monitoring.
    -   **Dependency Cleanup**: `apt-get install ... && rm -rf /var/lib/apt/lists/*` keeps the layer size down.
-   **Environment**: Sets `MLFLOW_TRACKING_URI` to use the local SQLite database within the container.
-   **Entrypoint**: Launches the Flask application on port 5000.

### 5.3 Kubernetes Architecture
The application is designed to be cloud-native:
\```mermaid
graph TD
    LB[Load Balancer] --> Svc[K8s Service]
    Svc --> Pod1[Replica 1]
    Svc --> Pod2[Replica 2]
    Svc --> Pod3[Replica 3]
    subgraph "Pod Internals"
        Pod1 --> Flask[Flask App]
        Flask --> Model[Loaded Model]
    end
\```
-   **High Availability Deployment**:
    -   **Replicas**: `2` (Ensures zero downtime if one pod fails).
    -   **Resource Limits**:
        -   **Requests**: `128Mi` RAM, `100m` CPU (Guaranteed minimums).
        -   **Limits**: `512Mi` RAM, `500m` CPU (Prevents “noisy neighbor” issues).
-   **Self-Healing (Probes)**:
    -   **Liveness**: Checks `/health` every 20s. Restarts pod if dead.
    -   **Readiness**: Checks `/health` every 10s. Removes pod from LoadBalancer if not ready to serve traffic.
-   **Service Exposure**:
    -   **Type**: `LoadBalancer`.
    -   **Port Mapping**: External Port `5000` $\rightarrow$ Container Port `5000`.
-   **Observability**:
    -   **ServiceMonitor**: Configured to scrape metrics from `/metrics` every `15s` for Prometheus integration.

---

## 6. Project Resources

-   **Code Repository**: [https://github.com/Ezaz13/MLOPS-Heart-Disease.git](https://github.com/Ezaz13/MLOPS-Heart-Disease.git)
-   **Application Demo Video**: [Watch Recording](https://wilpbitspilaniacin0-my.sharepoint.com/:v:/g/personal/2024aa05083_wilp_bits-pilani_ac_in/IQByKQZjA01pTZDbOenZWesFAWihpS3Pcps-bkGT9a37-tg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=2EyaNt)