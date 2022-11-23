"""FastAPI app that includes pydantic data model for expected incoming
inference candidate schema, a greeting response for root get requests, and a
model call for post requests at the /predict endpoint.
"""

import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Instantiate the app
app = FastAPI()


# Use of hyphenated feature names in the Respondent class definition below
# results in `SyntaxError: illegal target for annotation`. We use a function
# defined here, later added to the class
def to_hyphen(string: str) -> str:
    return string.replace("-", "_")


# Declare example of data inference artifact is expecting using pydantic
class Respondent(BaseModel):
    age: int = Field(example=40)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=12285)
    education: str = Field(example=9)
    education_num: int = Field(example=10)
    marital_status: str = Field(example="Married-civ-spouse")
    occupation: str = Field(example="Craft-repair")
    relatonship: str = Field(example="Own-child")
    race: str = Field(example="Black")
    sex: str = Field(example="Female")
    capital_gain: int = Field(example=50)
    capital_loss: int = Field(example=1000)
    hours_per_week: int = Field(example=35)
    native_country: str = Field(example="United-States")
    salary: int = Field(example=">50k")

    class Config:
        alias_generator = to_hyphen


# Declare api response expectation
class Response(Respondent):
    prediction: bool


# Load model
clf = pickle.load(open("model/models/model.pkl", "rb"))


@app.get("/")
async def welcome():
    return {"greeting": "Welcome to the Census Income Predictor"}


# @app.post("/predict/")
# async def predict(respondent: Respondent):
#     return None
