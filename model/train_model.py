"""Script to train machine learning model.
"""
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model

# Ingest data
data = pd.read_csv("../data/census.csv")

# Test train split
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
)
# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model, open("model.pkl", "wb"))
