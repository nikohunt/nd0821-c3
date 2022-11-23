from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

fake_positive_respondent = {
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

fake_negative_respondent = {
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


def test_post_predict_success():
    """Tests that the post endpoint returns a success code upon POST"""
    r = client.post(
        "/predict/",
        json=fake_positive_respondent,
    )
    assert r.status_code == 200
