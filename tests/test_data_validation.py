import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_raw():
    return pd.read_csv("/Users/caroletene/California-Housing/datasets/housing.csv")


def test_expected_columns_exist():
    df = load_raw()

    expected_cols = [
        "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "median_income", "median_house_value"
    ]

    for col in expected_cols:
        assert col in df.columns


def test_target_reasonable_range():
    df = load_raw()

    # California housing values typically between 0 and 500000+
    assert df["median_house_value"].min() > 0
    assert df["median_house_value"].max() < 1000000


def test_numeric_ranges():
    df = load_raw()

    assert df["median_income"].min() >= 0
    assert df["housing_median_age"].max() < 100