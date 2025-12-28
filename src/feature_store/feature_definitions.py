"""
Feature Definitions for the Customer Churn Prediction Model.

This module provides a centralized registry for all engineered features,
including their metadata, such as descriptions, data types, and sources.
This approach serves as a lightweight, custom feature store.
"""

# A list of dictionaries, where each dictionary defines a feature's metadata.
FEATURE_DEFINITIONS = [
    # --- Original Features (Cleaned & Prepared) ---
    {
        "name": "gender",
        "description": "The customer's gender.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["demographics"]
    },
    {
        "name": "SeniorCitizen",
        "description": "Whether the customer is a senior citizen (1) or not (0).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["demographics"]
    },
    {
        "name": "Partner",
        "description": "Whether the customer has a partner (Yes) or not (No).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["demographics"]
    },
    {
        "name": "Dependents",
        "description": "Whether the customer has dependents (Yes) or not (No).",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["demographics"]
    },
    {
        "name": "tenure",
        "description": "Number of months the customer has stayed with the company.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["service_usage"]
    },
    {
        "name": "MonthlyCharges",
        "description": "The amount charged to the customer monthly.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["financial"]
    },
    {
        "name": "TotalCharges",
        "description": "The total amount charged to the customer.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["financial"]
    },
    {
        "name": "Churn",
        "description": "Whether the customer churned (Yes) or not (No). This is the target variable.",
        "source": "Original Dataset",
        "version": 1.0,
        "data_type": "Categorical",
        "tags": ["target"]
    },

    # --- Engineered Features ---
    {
        "name": "charge_ratio",
        "description": "Ratio of MonthlyCharges to TotalCharges. Helps identify customers with high recent spending relative to their history.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["financial", "derived"]
    },
    {
        "name": "num_optional_services",
        "description": "The total number of optional services a customer has subscribed to (e.g., OnlineSecurity, TechSupport).",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["service_usage", "aggregated"]
    },
    {
        "name": "tenure_charge_interaction",
        "description": "Product of tenure and MonthlyCharges. Captures the combined effect of loyalty and spending.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Numerical",
        "tags": ["financial", "derived"]
    },
    {
        "name": "is_high_value",
        "description": "A binary flag indicating if a customer is 'high-value' (top 25% tenure and monthly charges).",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["financial", "derived"]
    },

    # --- One-Hot Encoded Features from Tenure Groups ---
    {
        "name": "tenure_q1",
        "description": "One-hot encoded feature representing the first quartile of tenure.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["service_usage", "derived", "one-hot"]
    },
    {
        "name": "tenure_q2",
        "description": "One-hot encoded feature representing the second quartile of tenure.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["service_usage", "derived", "one-hot"]
    },
    {
        "name": "tenure_q3",
        "description": "One-hot encoded feature representing the third quartile of tenure.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["service_usage", "derived", "one-hot"]
    },
    {
        "name": "tenure_q4",
        "description": "One-hot encoded feature representing the fourth quartile of tenure.",
        "source": "Feature Engineering Pipeline",
        "version": 1.0,
        "data_type": "Binary",
        "tags": ["service_usage", "derived", "one-hot"]
    }
]

def get_feature_definitions():
    """Returns the list of all feature definitions."""
    return FEATURE_DEFINITIONS

def get_feature_names():
    """Returns a list of all feature names."""
    return [f["name"] for f in FEATURE_DEFINITIONS]
