#  California Housing – End-to-End MLOps Pipeline

This project implements a complete Machine Learning Operations (MLOps) pipeline for predicting California housing prices using a regression model. It demonstrates how to take a raw dataset through the full ML lifecycle including training, testing, deployment readiness, and monitoring.
---
# Project Structure
src/            → Training + drift monitoring scripts  
tests/          → Unit + data + model tests  
datasets/       → Raw data (tracked with DVC)  
configs/        → YAML configuration files  
reports/        → Drift reports  
.github/        → CI/CD workflows  

# Key Components
## Machine Learning

.Model: Linear Regression
.Task: Predict median_house_value
.Preprocessing:
    .Missing value handling (median for numeric, mode for .categorical)
    .One-hot encoding for categorical variables
    .Train/test split

---

## Experiment Tracking (MLflow)

Each training run is tracked using :contentReference[oaicite:0]{index=0}:

Tracked metrics:
- RMSE
- MAE
- R² score

Tracked artifacts:
- trained model
- configuration file

---

## Testing (Pytest)
   . Unit tests for preprocessing functions
   .Data validation tests (columns, ranges, structure)
   .Model tests (prediction shape + performance checks)

  => to  Run tests:
     pytest tests/ -v

## Version Control (DVC)

Dataset is tracked using :contentReference[oaicite:1]{index=1}:
- Raw data is NOT stored in Git
- Only DVC metadata is committed
- Enables reproducible data pipelines

---

## CI/CD Pipeline (GitHub Actions)

The pipeline includes two jobs:

### 1. Test Job
- Installs dependencies
- Runs full pytest suite

### 2. Train Job
- Runs only if tests pass
- Trains the model
- Logs metrics to MLflow
- Fails if performance threshold is not met

Trigger:
- Push to `main`
- Pull request to `main`

---
## Data Drift Monitoring

Uses Evidently to:

- Compare training vs production-like data
- Detect feature drift
- Generate HTML report
- Exit with error code if drift exceeds threshold

# How to run

#### Install dependencies
        pip install -r requirements.txt
#### Train model
        python src/train.py
#### Run tests
        pytest tests/ -v
#### Run drift monitoring
        python src/monitor_drift.py


# Pipeline Flow
Raw Data → DVC → Preprocessing → Model Training → MLflow Tracking → CI/CD → Drift Monitoring