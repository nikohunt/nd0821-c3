from fastapi import FastAPI
from pydantic import BaseModel

# Instantiate the app.
app = FastAPI()


# Setup pytdantic body value
class Value(BaseModel):
    value: int


@app.get("/")
async def welcome():
    return {"greeting": "Welcome to the Census Income Predictor"}
