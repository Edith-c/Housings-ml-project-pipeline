import numpy as np
from src.train import load_and_prepare_data, build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_model_prediction_shape():
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model({"model_type": "linear_regression"})
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    assert len(preds) == len(y_test)


def test_model_minimum_performance():
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model({"model_type": "linear_regression"})
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    # Very low bar just to ensure model works
    assert score > 0.3


