from ml.data import process_data
from ml.model import compute_model_metrics, inference


def category_slice(test, model, cat_features, encoder, lb, features):
    """Outputs the text file slice_output.txt to model/slices that gives
    performance metrics on model slices for a given list of features.

    Args:
        test (pd.DataFrame): Test dataset that will be filtered to obtain slice
        model (sklearn.Model): Model for slice evaluation
        cat_features (list): Categorical features in the model
        encoder (sklearn.Transformer): Saved OneHotEncoder for feature
            engineering
        lb (sklearn.Transformer): Saved label binarizer for feature engineering
        features (list): Feature/s under examination that will appear in
            slice_output.txt
    """
    with open("model/slices/slice_output.txt", "w") as f:
        for feature in features:
            print(f"Feature: {feature}", file=f)
            for cat in test[feature].unique():
                print(f"    Category: {cat}", file=f)
                test_filtered = test[test[feature] == cat]
                X_test, y_test, encoder, lb = process_data(
                    test_filtered,
                    categorical_features=cat_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb,
                )
                preds = inference(model, X_test)
                precision, recall, fbeta = compute_model_metrics(y_test, preds)
                print(f"        Precision: {precision:.2f}", file=f)
                print(f"        Recall: {recall:.2f}", file=f)
                print(f"        Fbeta: {fbeta:.2f}", file=f)
