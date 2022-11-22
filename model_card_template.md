# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is developed by Nikolas Hunt and v1.0.0 was trained 22nd November 2022. The model is an [SKLearn SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) which is SciKit-Learn's implementation of a Stochastic Gradient Descent classifier.

The model was trained using the default parameters of:

- loss='hinge'
- penalty='l2'
- alpha=0.0001
- l1_ratio=0.15
- fit_intercept=True
- max_iter=1000
- tol=0.001
- shuffle=True
- verbose=0
- epsilon=0.1
- n_jobs=None
- random_state=None
- learning_rate='optimal'
- eta0=0.0
- power_t=0.5
- early_stopping=False
- validation_fraction=0.1
- n_iter_no_change=5
- class_weight=None
- warm_start=False
- average=False

## Intended Use

During development, the use case envisioned was to demonstrate the implementation of a model for FastAPI inferencing and deployment to heroku.

The model itself takes census data and predicts whether the salary of the respondent is over $50,000 or not. As such, given new census data of a statistically similar nature, the model could infer the salary.

Primary users will be the project evaluator.

All use cases apart from the evaluation for a Udacity project are out of scope.

## Training Data

The model was trained on the UCI Machine Learning Repository [_Census Income_](https://archive.ics.uci.edu/ml/datasets/census+income) dataset.

The dataset was extracted by Barry Becker from the 1994 US Census and was donated to UCI May 1st, 1996.

There are 26,048 instances in the training dataset, representing 80% of the original 32,561 in data/census.csv, and 14 attributes.

Distributions over the attributes can be found in the eda folder of this project.

## Evaluation Data

There are 6,513 instances in the test dataset.

## Metrics

Being a binary classifier, precision, recall, and f1 scores are all used. Although the code does reference f-beta, the current implementation uses a beta of 1, which makes the calculation equivalent to f1.

F1 is the harmonic mean of precision and recalll. Whereas the regular mean treats all values equally, the harmonic mean gives much more weight ot low values. As a result, the classifier will only get a high F1 score if both recall and precision are high.

- Precision: 0.56
- Recall: 0.34
- F-beta: 0.42

## Ethical Considerations

Although this is census respondent data, it is publicly available and does not represent peronsallly identifiable information.

It should be noted that model slicing found the following discrepancies for socio-economic groups:

- The 'Other' category for race strongly outperformed all other categories when measuring by f-beta
- Respondents in the 'Without-pay' category of the workclass feature had an f-beta of zero, indicating that the model did not train enough on this slice and cannot be relied on for such respondents when infering.
- Widowed respondents shows a poor f1-beta score.

## Caveats and Recommendations

The above notes can be taken as recommending that the model not be used for any purpose outside of being an exercise for model deployment.

Should a model solution be required, at the very least, hyperparameter tuning and scaling would be the minimum steps required to improve performance as it is currently unacceptably low.
