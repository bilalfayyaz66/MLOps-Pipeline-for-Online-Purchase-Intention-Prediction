# MLOps Pipeline for Online Purchase Intention Prediction

This repository contains an end-to-end MLOps project for predicting whether an online shopping session will result in a purchase. The project uses a real e-commerce dataset, trains multiple machine learning models, tracks experiments with MLflow, serves the best model through a FastAPI application, packages the application using Docker, monitors the running service with Prometheus and Grafana, and automates testing/building through GitHub Actions.

The goal of this project is not only to build a machine learning model, but to demonstrate a complete production-style machine learning workflow.

---

## Project Overview

Machine learning models are often developed in notebooks or scripts, but real-world systems require more than model accuracy. A practical ML system should be reproducible, containerized, testable, deployable, and observable after deployment.

This project implements a complete MLOps pipeline for an e-commerce classification problem. The system predicts whether a user session on an online shopping website will generate revenue.

The pipeline includes:

- Data loading and preprocessing
- Model training and evaluation
- MLflow experiment tracking
- Model packaging
- FastAPI-based prediction API
- Docker containerization
- Prometheus metrics collection
- Grafana dashboard visualization
- GitHub Actions CI/CD pipeline

---

## Research Problem

The main research problem addressed in this project is:

> How can an end-to-end MLOps pipeline improve the reproducibility, deployment readiness, and observability of an online purchase intention prediction system?

---
## Research Component

This project follows Track-II: Technical Research (Implementation + Improvement).

The research component includes:

clearly defined research problem
8–10 research references in the IEEE paper
research questions
model evaluation metrics
operational monitoring metrics
experimental validation
performance analysis
## Research Questions

The project is guided by the following research questions:

1. Which model performs best for online purchase intention prediction among Logistic Regression, Random Forest, and XGBoost?
2. Can an end-to-end MLOps pipeline be implemented using open-source tools such as MLflow, Docker, Prometheus, Grafana, and GitHub Actions?
3. How effectively can the deployed machine learning model be monitored in terms of availability, prediction traffic, and inference latency?

---

## Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset**.

The dataset contains session-level information from an e-commerce website. Each row represents a user session, and the target variable indicates whether that session resulted in revenue.

### Target Variable

| Column | Description |
|---|---|
| `Revenue` | Binary target indicating whether the session resulted in a purchase |

### Feature Examples

The dataset contains numerical and categorical features, including:

- `Administrative`
- `Administrative_Duration`
- `Informational`
- `Informational_Duration`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `ExitRates`
- `PageValues`
- `SpecialDay`
- `Month`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`
- `VisitorType`
- `Weekend`

---

## Tech Stack

| Area | Tool / Technology |
|---|---|
| Programming Language | Python |
| ML Models | Logistic Regression, Random Forest, XGBoost |
| ML Library | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Model Serialization | Joblib |
| Containerization | Docker |
| Multi-Service Orchestration | Docker Compose |
| Metrics Collection | Prometheus |
| Dashboard / Visualization | Grafana |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Development Environment | Windows + WSL Ubuntu |

---


## Experimental Analysis Summary

The experiments show that XGBoost performed best among the evaluated models.

Compared with Logistic Regression:

F1-score improved from 0.4814 to 0.6570
Recall improved from 0.3560 to 0.5942
ROC-AUC improved from 0.8876 to 0.9305

The deployed API also showed strong operational performance:

API remained available during testing
Prometheus successfully collected metrics
Grafana displayed real-time monitoring values
Average model latency was approximately 7.87 ms


## Model Perfromace 
Model Performance
Model	                      Accuracy               	      Precision              Recall	              F1-score                	ROC-AUC
Logistic Regression	        0.8812                       	 0.7432	              0.3560	                0.4814	                0.8876
Random Forest			        0.8974                          0.7416                0.5183	                0.6102	                0.9194
XGBoost	                    0.9039	                         0.7346	              0.5942	                0.6570	                0.9305



## System Architecture

The project follows an end-to-end MLOps architecture.

```text
Dataset
   |
   v
Data Preprocessing
   |
   v
Model Training
   |
   +--> Logistic Regression
   +--> Random Forest
   +--> XGBoost
   |
   v
Model Evaluation
   |
   v
MLflow Experiment Tracking
   |
   v
Best Model Selection
   |
   v
Serialized Model Artifact
   |
   v
FastAPI Prediction Service
   |
   v
Docker Container
   |
   +--> Prometheus scrapes /metrics
   |
   v
Grafana Dashboard

GitHub Repository
   |
   v
GitHub Actions CI/CD
   |
   +--> Run tests
   +--> Build Docker image

![Architecture Diagram](reports/architecture.png)




