"""Script to train machine learning model.
"""
import logging
import pickle

import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go():
    # Ingest data
    logging.info("INFO      Ingesting data")
    data = pd.read_csv("data/census.csv", skipinitialspace=True)
    logging.info("SUCCESS   Data ingested")

    # Test train split
    logging.info("INFO      Making test train split")
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    logging.info("SUCCESS   Data split")

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

    logging.info("INFO      Feature engineering")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data with the process_data function.
    logging.info("INFO      Ingesting data")
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    logging.info("SUCCESS   Feature engineering")

    # Train model
    logging.info("INFO      Training model")
    model = train_model(X_train, y_train)
    logging.info("SUCCESS   Model trained")

    # Compute metrics
    logging.info("INFO      Computing model metrics")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logging.info(f"INFO      Precision: {precision:.2f}")
    logging.info(f"INFO      Recall: {recall:.2f}")
    logging.info(f"INFO      Fbeta: {fbeta:.2f}")
    logging.info("SUCCESS   Model metrics")

    # Save model and encoder
    pickle.dump(model, open("model/models/model.pkl", "wb"))
    pickle.dump(encoder, open("model/encoders/encoder.pkl", "wb"))
    logging.info("SUCCESS   Model and encoder saved")


if __name__ == "__main__":
    go()
