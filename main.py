"""FastAPI app that includes pydantic data model for expected incoming
inference candidate schema, a greeting response for root get requests, and a
model call for post requests at the /predict endpoint
"""
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

# Instantiate the app
app = FastAPI()


def to_hyphen(string: str) -> str:
    """Use of hyphenated feature names in the Respondent class definition below
    results in `SyntaxError: illegal target for annotation`. We use a function
    defined here, later added to the class

    Args:
        string (str): String featuring underscores

    Returns:
        str: String with underscores replaced by hyphens
    """
    return string.replace("_", "-")


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

    class Config:
        alias_generator = to_hyphen


# Declare api response expectation
class Response(Respondent):
    prediction: bool


# Load model and encoders
clf = pickle.load(open("model/models/model.pkl", "rb"))
encoder = pickle.load(open("model/encoders/encoder.pkl", "rb"))


def pydantic_model_to_df(model_instance: Respondent) -> pd.DataFrame:
    """Converts pydantic model instance into a Pandas DataFrame which is
    expected by the model

    Args:
        model_instance (Respondent): Pydantic class inherited from BaseModel

    Returns:
        pd.DataFrame: Pandas DataFrame ready for inference
    """
    return pd.DataFrame([jsonable_encoder(model_instance)])


@app.get("/")
async def welcome():
    return {"greeting": "Welcome to the Census Income Predictor"}


@app.post("/predict/", response_model=Response)
async def predict(respondent: Respondent):
    # Convert body to pandas
    df_instance = pydantic_model_to_df(respondent)

    # Encoder for categorical features
    encoder.transform(df_instance)

    # Run inference
    prediction = clf.predict(df_instance).tolist()[0]

    # Construct api response
    response = respondent.dict(by_alias=True)
    response.update({"Predictions": prediction})
    return response
