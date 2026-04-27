import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from xgboost import XGBClassifier

from src.data import load_data, split_data, build_preprocessor


def get_model(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss",
            random_state=42
        )

    raise ValueError(f"Unknown model: {model_name}")


def train(data_path: str, model_name: str):
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    model = get_model(model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    mlflow.set_experiment("purchase-intention-mlops")

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("rows", len(df))
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(pipeline, "model")

        joblib.dump(pipeline, "models/model.joblib")

        print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/online_shoppers_intention.csv")
    parser.add_argument("--model", default="random_forest")
    args = parser.parse_args()

    train(args.data, args.model)