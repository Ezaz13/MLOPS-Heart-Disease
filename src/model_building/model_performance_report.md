# Model Performance Report

*Report generated on: 2025-12-29 12:46:36
*MLflow Experiment: 'Heart Disease Prediction'

## Logistic Regression

- **MLflow Run ID**: `18279d95d8b64c1da0c46cee5ddd0742`
- **Best Params**: `{'C': 0.1, 'solver': 'liblinear'}`
- **CV Accuracy (Mean)**: 0.8345
- **Test Accuracy**: 0.8852
- **Precision**: 0.8387
- **Recall**: 0.9286
- **F1-Score**: 0.8814
- **ROC-AUC**: 0.9621

### Classification Report

```
              precision    recall  f1-score   support

           0       0.93      0.85      0.89        33
           1       0.84      0.93      0.88        28

    accuracy                           0.89        61
   macro avg       0.89      0.89      0.89        61
weighted avg       0.89      0.89      0.89        61

```

## Random Forest

- **MLflow Run ID**: `782a6aafd1d44b91ac72188107a8b975`
- **Best Params**: `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}`
- **CV Accuracy (Mean)**: 0.8136
- **Test Accuracy**: 0.8852
- **Precision**: 0.8387
- **Recall**: 0.9286
- **F1-Score**: 0.8814
- **ROC-AUC**: 0.9394

### Classification Report

```
              precision    recall  f1-score   support

           0       0.93      0.85      0.89        33
           1       0.84      0.93      0.88        28

    accuracy                           0.89        61
   macro avg       0.89      0.89      0.89        61
weighted avg       0.89      0.89      0.89        61

```

