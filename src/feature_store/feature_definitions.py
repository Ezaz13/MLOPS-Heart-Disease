"""
Feature Definitions for the Heart Disease Prediction Model.

This module provides a centralized registry for all engineered features,
including their metadata, such as descriptions, data types, and sources.
This approach serves as a lightweight, custom feature store.
"""

# A list of dictionaries, where each dictionary defines a feature's metadata.
FEATURE_DEFINITIONS = [
    # --- Original Features (Cleaned & Prepared) ---
    {
        "name": "age",
        "description": "Age in years.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["demographics"]
    },
    {
        "name": "sex",
        "description": "Sex (1 = male; 0 = female).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["demographics"]
    },
    {
        "name": "cp",
        "description": "Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["clinical"]
    },
    {
        "name": "trestbps",
        "description": "Resting blood pressure (in mm Hg on admission to the hospital).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["clinical"]
    },
    {
        "name": "chol",
        "description": "Serum cholestoral in mg/dl.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["clinical"]
    },
    {
        "name": "fbs",
        "description": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["clinical"]
    },
    {
        "name": "restecg",
        "description": "Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["clinical"]
    },
    {
        "name": "thalach",
        "description": "Maximum heart rate achieved.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["clinical"]
    },
    {
        "name": "exang",
        "description": "Exercise induced angina (1 = yes; 0 = no).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["clinical"]
    },
    {
        "name": "oldpeak",
        "description": "ST depression induced by exercise relative to rest.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["clinical"]
    },
    {
        "name": "slope",
        "description": "The slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["clinical"]
    },
    {
        "name": "ca",
        "description": "Number of major vessels (0-3) colored by flourosopy.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["clinical"]
    },
    {
        "name": "thal",
        "description": "Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["clinical"]
    },
    {
        "name": "target",
        "description": "Diagnosis of heart disease (1 = presence, 0 = absence).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["target"]
    },

    # --- Engineered Features ---
    {
        "name": "rate_pressure_product",
        "description": "Product of Max Heart Rate (thalach) and Resting Blood Pressure (trestbps). Measure of cardiac work.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["clinical", "derived"]
    },
    {
        "name": "chol_fbs_interaction",
        "description": "Interaction between Cholesterol and Fasting Blood Sugar.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["clinical", "derived"]
    },
    {
        "name": "is_high_risk",
        "description": "A binary flag indicating high risk based on ST depression (oldpeak) and major vessels (ca).",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["clinical", "derived"]
    },

    # --- One-Hot Encoded Features from Age Groups ---
    {
        "name": "age_q1",
        "description": "One-hot encoded feature representing the first quartile of age.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["demographics", "derived", "one-hot"]
    },
    {
        "name": "age_q2",
        "description": "One-hot encoded feature representing the second quartile of age.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["demographics", "derived", "one-hot"]
    },
    {
        "name": "age_q3",
        "description": "One-hot encoded feature representing the third quartile of age.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["demographics", "derived", "one-hot"]
    },
    {
        "name": "age_q4",
        "description": "One-hot encoded feature representing the fourth quartile of age.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["demographics", "derived", "one-hot"]
    }
]

def get_feature_definitions():
    """Returns the list of all feature definitions."""
    return FEATURE_DEFINITIONS

def get_feature_names():
    """Returns a list of all feature names."""
    return [f["name"] for f in FEATURE_DEFINITIONS]
