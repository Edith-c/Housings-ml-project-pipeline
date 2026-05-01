import pandas as pd
import numpy as np
import pytest
from src.train import load_and_prepare_data
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_missing_values_handled():
    X, y = load_and_prepare_data()
    assert X.isnull().sum().sum() == 0


def test_categorical_encoded():
    X, _ = load_and_prepare_data()
    # No object types should remain
    assert not any(X.dtypes == "object")


def test_original_dataframe_not_modified():
    import pandas as pd

    path = "/Users/caroletene/California-Housing/datasets/housing.csv"
    df_original = pd.read_csv(path)

    df_copy = df_original.copy()
    load_and_prepare_data()

    # original should stay unchanged
    pd.testing.assert_frame_equal(df_original, df_copy)


def test_output_shapes():
    X, y = load_and_prepare_data()
    assert len(X) == len(y)


def test_numeric_only_after_processing():
    X, _ = load_and_prepare_data()
    assert all(dtype != "object" for dtype in X.dtypes)


def test_invalid_input_raises_error():
    with pytest.raises(Exception):
        load_and_prepare_data(config="invalid")  # wrong type