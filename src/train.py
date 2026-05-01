import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─── Configs ─────────────────────────────────────
configs = [
    {"model_type": "linear_regression", "test_size": 0.2, "random_state": 42},
    {"model_type": "linear_regression", "test_size": 0.2, "random_state": 1},
    {"model_type": "linear_regression", "test_size": 0.3, "random_state": 42},
    {"model_type": "linear_regression", "test_size": 0.25, "random_state": 7},
    {"model_type": "linear_regression", "test_size": 0.15, "random_state": 99},
]


# ─── Load + Prepare Data ─────────────────────────
def load_and_prepare_data():
    path = "/Users/caroletene/California-Housing/datasets/housing.csv"
    df = pd.read_csv(path)

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Fill numeric missing with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical missing with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Split X / y
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    return X, y


# ─── Build Model ─────────────────────────────────
def build_model(config):
    return LinearRegression()


# ─── Run Experiment ──────────────────────────────
def run_experiment(config):

    mlflow.set_experiment("median_house_value-prediction")

    with mlflow.start_run():

        # Load data FIRST ✅
        X, y = load_and_prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["test_size"],
            random_state=config["random_state"]
        )

        # Build + Train
        model = build_model(config)
        print("\nTraining Linear Regression...")
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log params
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("n_rows", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save config
        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        mlflow.log_artifact(config_path)
        os.remove(config_path)

        # Print results
        print("\n" + "="*40)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        print("="*40)

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")

        return run_id


# ─── Main ───────────────────────────────────────
if __name__ == "__main__":
    for i, config in enumerate(configs):
        print(f"\nRunning experiment {i+1}/5...")
        run_experiment(config)