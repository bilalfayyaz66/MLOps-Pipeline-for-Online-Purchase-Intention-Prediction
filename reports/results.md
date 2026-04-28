# Experimental Results

## Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8812 | 0.7432 | 0.3560 | 0.4814 | 0.8876 |
| Random Forest | 0.8974 | 0.7416 | 0.5183 | 0.6102 | 0.9194 |
| XGBoost | 0.9039 | 0.7346 | 0.5942 | 0.6570 | 0.9305 |

## Best Model

XGBoost was selected as the best model because it achieved the highest accuracy, recall, F1-score, and ROC-AUC.

## Monitoring Results

| Metric | Observed Value |
|---|---:|
| API Up Status | 1 |
| Total Model Predictions | 50+ |
| Positive Purchase Predictions | 0 |
| Average Model Latency | ~7.87 ms |

## Tools Used

- MLflow for experiment tracking
- FastAPI for model serving
- Docker for containerization
- Prometheus for metrics collection
- Grafana for dashboard visualization
- GitHub Actions for CI/CD automation
