import pandas as pd
import pytest


@pytest.fixture(scope="session")
def data():
    return pd.read_csv("data/census.csv", skipinitialspace=True)


def test_column_presence_and_type(data):
    """Tests that cleaned csv file has expected columns and types.

    Args:
        data (pd.DataFrame): Dataset for testing
    """

    required_columns = {
        "age": pd.api.types.is_int64_dtype,
        "workclass": pd.api.types.is_object_dtype,
        "fnlgt": pd.api.types.is_int64_dtype,
        "education": pd.api.types.is_object_dtype,
        "education-num": pd.api.types.is_int64_dtype,
        "marital-status": pd.api.types.is_object_dtype,
        "occupation": pd.api.types.is_object_dtype,
        "relationship": pd.api.types.is_object_dtype,
        "race": pd.api.types.is_object_dtype,
        "sex": pd.api.types.is_object_dtype,
        "capital-gain": pd.api.types.is_int64_dtype,
        "capital-loss": pd.api.types.is_int64_dtype,
        "hours-per-week": pd.api.types.is_int64_dtype,
        "native-country": pd.api.types.is_object_dtype,
        "salary": pd.api.types.is_object_dtype,
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def test_race_names(data):

    known_race_names = [
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
    ]
    print(data.race.values)
    assert set(known_race_names).issuperset(set(data["race"].values))
