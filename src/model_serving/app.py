import os
import pathlib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.sklearn

# -------------------------------------------------
# Paths & MLflow (DB BACKEND ONLY CHANGE)
# -------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

# ❌ OLD (filesystem – deprecated)
# mlflow.set_tracking_uri("file:///mlruns")

# ✅ NEW (SQLite database backend)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/HeartDiseaseModel/latest"

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
model = None

# Categorical features used during training
CATEGORICAL_FEATURES = ["thal"]

# -------------------------------------------------
# Load model
# -------------------------------------------------
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        app.logger.info(f"Model loaded from {MODEL_URI}")
    except Exception as e:
        model = None
        app.logger.error(f"Failed to load model: {e}")

# -------------------------------------------------
# UI
# -------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# -------------------------------------------------
# Health
# -------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI
    })

# -------------------------------------------------
# Prediction
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        df = pd.DataFrame([data])

        # ---------------------------------------------------------
        # 1. Data Mapping (Form values -> Model training standards)
        # ---------------------------------------------------------
        if 'cp' in df.columns:
            df['cp'] = df['cp'].map({0: 1, 1: 2, 2: 3, 3: 4})

        if 'slope' in df.columns:
            df['slope'] = df['slope'].map({0: 1, 1: 2, 2: 3})

        if 'thal' in df.columns:
            df['thal'] = df['thal'].map({1: 3, 2: 6, 3: 7})

        # ---------------------------------------------------------
        # 2. Feature Engineering
        # ---------------------------------------------------------
        df['rate_pressure_product'] = df['trestbps'] * df['thalach']
        df['chol_fbs_interaction'] = df['chol'] * df['fbs']
        df['is_high_risk'] = 0

        # ---------------------------------------------------------
        # 3. One-Hot Encoding & Alignment
        # ---------------------------------------------------------
        cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df_processed = pd.get_dummies(df, columns=cat_cols)

        if hasattr(model, "feature_names_in_"):
            df_processed = df_processed.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        preds = model.predict(df_processed)
        probs = model.predict_proba(df_processed)[:, 1]

        return jsonify([{
            "prediction": int(preds[0]),
            "confidence": float(probs[0])
        }])

    except Exception as e:
        app.logger.error(f"Inference failed: {e}")
        return jsonify({
            "error": "Inference failed",
            "details": str(e)
        }), 400

# -------------------------------------------------
# Entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
