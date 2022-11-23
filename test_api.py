import pytest
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


@pytest.fixture
def fake_positive_respondent():
    return {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 12285,
        "education": "9",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 50,
        "capital-loss": 1000,
        "hours-per-week": 35,
        "native-country": "United-States",
    }


@pytest.fixture
def fake_negative_respondent():
    return {
        "age": 17,
        "workclass": "?",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "Mexican",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 1,
        "native-country": "?",
    }


def test_get_root_success():
    """Tests that the root endpoint returns a success code upon GET"""
    r = client.get("/")
    assert r.status_code == 200


def test_get_root_content():
    """Tests that the root endpoint returns expected json content in
    response"""
    r = client.get("/")
    assert r.json() == {"greeting": "Welcome to the Census Income Predictor"}


def test_post_predict_success(fake_positive_respondent):
    """Tests that the post endpoint returns a success code upon POST"""
    r = client.post(
        "/predict/",
        json=fake_positive_respondent,
    )
    assert r.status_code == 200


def test_post_predict_postive_content(fake_positive_respondent):
    """Tests that the post endpoint returns expected positive outcome"""
    r = client.post(
        "/predict/",
        json=fake_positive_respondent,
    )
    fake_positive_respondent["prediction"] = 1
    assert r.json() == fake_positive_respondent


def test_post_predict_negative_content(fake_negative_respondent):
    """Tests that the post endpoint returns expected negative outcome"""
    r = client.post(
        "/predict/",
        json=fake_negative_respondent,
    )
    fake_negative_respondent["prediction"] = 0
    assert r.json() == fake_negative_respondent
