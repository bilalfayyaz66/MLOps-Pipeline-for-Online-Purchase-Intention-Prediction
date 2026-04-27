import time
import joblib
import pandas as pd

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import PredictionInput


app = FastAPI(title="Purchase Intention Prediction API")

model = joblib.load("models/model.joblib")

prediction_counter = Counter(
    "model_predictions_total",
    "Total number of model predictions"
)

purchase_prediction_counter = Counter(
    "purchase_predictions_total",
    "Total number of positive purchase predictions"
)

prediction_latency = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency in seconds"
)


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "purchase-intention-api"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    start_time = time.time()

    df = pd.DataFrame([input_data.model_dump()])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    prediction_counter.inc()

    if prediction == 1:
        purchase_prediction_counter.inc()

    prediction_latency.observe(time.time() - start_time)

    return {
        "purchase_prediction": prediction,
        "purchase_probability": probability
    }


Instrumentator().instrument(app).expose(app)
