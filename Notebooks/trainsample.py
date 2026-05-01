import pandas as pd
df = pd.read_csv('/Users/caroletene/California-Housing/datasets/housing.csv')
print(df.columns.tolist())
print(df.shape)
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
print(categorical_cols)
print(df.ocean_proximity.nunique())
df_cat = pd.get_dummies(df, columns=categorical_cols, drop_first = True)
print(df_cat)

import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import subprocess
import sys
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
# ─── Helper: get DVC version ───────────────────────────────────────────────

def get_dvc_version():
    try:
        result = subprocess.run(
            ["dvc", "status", "--json"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
configs = [
    {"model_type": "linear_regression", "test_size": 0.2, "random_state": 42},
    {"model_type": "linear_regression", "test_size": 0.2, "random_state": 1},
    {"model_type": "linear_regression", "test_size": 0.3, "random_state": 42},
    {"model_type": "linear_regression", "test_size": 0.25, "random_state": 7},
    {"model_type": "linear_regression", "test_size": 0.15, "random_state": 99},
]

#============Load and prepare dataset===========

def load_and_prepare_data(config = None):
    """Load the california housing dataset and prepare it for training."""

    #path = config["datasets"]['path']
    path = '/Users/caroletene/California-Housing/datasets/housing.csv'
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("Filled missing values with median")

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # Fill categorical with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values handled: median (numeric), mode (categorical)")

    # Find categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

# Encode them
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features and target
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    return X, y, len(df), numeric_cols, categorical_cols

def build_model(config):
    """Create a Linear Regression model."""
    if config["model_type"] == "linear_regression":
        return LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
def run_experiment(config):
    mlflow.set_experiment("median_house_value-prediction")

    with mlflow.start_run():

        # ── Log config ──
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("test_size", config.get("test_size", 0.2))
        mlflow.log_param("random_state", config["random_state"])

        # ── Load data ──
        X, y, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)

        # ── Log dataset metadata ──
        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_numeric_features", len(numeric_cols))
        mlflow.log_param("n_categorical_features", len(categorical_cols))

        # ── Split data ──
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
            
        )


        # ── Train ──
        model = build_model(config)
        print(f"\nTraining Linear Regression...")
        model.fit(X_train, y_train)

        # ── Evaluate ──
        y_pred = model.predict(X_test)
       

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
   # metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # log model
        mlflow.sklearn.log_model(model, "model")

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        # ── Log the trained model as an artifact ──
        mlflow.sklearn.log_model(model, "model")

        # ── Log the config file as an artifact ──
        config_path = "config_snapshot.json"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        mlflow.log_artifact(config_path)
        os.remove(config_path)  # cleanup

        # ── Print results ──
        print(f"\n{'='*50}")
        print(f"Model: Linear Regression")

        print(f"RMSE:   {rmse:.4f}")
        print(f"MAE:    {mae:.4f}")
        print(f"R2:     {r2:.4f}")
        print(f"{'='*50}")

        run = mlflow.active_run()

        if run is not None:
            run_id = run.info.run_id
            print(f"\nMLflow Run ID: {run_id}")
            print("View this run in the UI: mlflow ui")
        else:
            run_id = None
            print("No active MLflow run found.")

        return run_id

    print(f"Run ID: {run_id}")
    print(f"Open in UI: http://127.0.0.1:5000/#/experiments/0/runs/{run_price}")
if __name__ == "__main__":
    for i, config in enumerate(configs):
        print(f"\nRunning experiment {i+1}/5...")
        run_experiment(config)

    