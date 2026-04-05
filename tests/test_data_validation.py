import pytest
from config.paths_config import *
import os
import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
)




@pytest.mark.skipif(
    not os.path.exists(TEST_FILE_PATH),
    reason="Dataset not available in CI",
)


def test_schema_and_types():
    df = pd.read_csv(TEST_FILE_PATH, parse_dates=["event_timestamp"])

    # Datetime columns
    assert is_datetime64_any_dtype(df["event_timestamp"])

    # Identifier
    assert is_integer_dtype(df["iris_id"])

    # Numeric features
    assert is_float_dtype(df["sepal_length"])
    assert is_float_dtype(df["sepal_width"])
    assert is_float_dtype(df["petal_length"])
    assert is_float_dtype(df["petal_width"])

    # Target label
    assert is_string_dtype(df["species"])

def test_no_missing_values():
    df = pd.read_csv(TEST_FILE_PATH)
    assert df.isnull().sum().sum() == 0

def test_value_ranges():
    df = pd.read_csv(TEST_FILE_PATH)

    assert df["sepal_length"].between(4.0, 8.0).all()
    assert df["sepal_width"].between(2.0, 4.5).all()
    assert df["petal_length"].between(1.0, 7.0).all()
    assert df["petal_width"].between(0.1, 3.0).all()

def test_species_values():
    df = pd.read_csv(TEST_FILE_PATH)
    allowed = {"setosa", "versicolor", "virginica"}
    assert set(df["species"].unique()).issubset(allowed)