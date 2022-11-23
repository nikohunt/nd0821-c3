"""Script to demonstrate POST request to the deployed herokuapp
"""

import logging

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

input_data = {
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

r = requests.post(
    url="https://desert-planet.herokuapp.com/predict", json=input_data
)

logging.info(f"SUCCESS:     Status code: {r.status_code}")
logging.info(f"INFO:        Server response: {r.json()}")
