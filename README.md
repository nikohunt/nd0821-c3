# Deploying a Machine Learning Model on Heroku with FastAPI

## Introduction

Using census data, as a data scientist, I have been given the task of predicitng whether a respondent's salary is greater than $50k or not, based on other features of that respondent.

This project trains that model and checks the performance of the data on slices of the dataset to understand fairness and bias.

The model has an api via FastAPI for inference, and the project includes code for deployment to Heroku, although [Heroku free product plans are to be removed November 28th, 2022](https://help.heroku.com/RSBRUH58/removal-of-heroku-free-product-plans-faq), at which point this code would require a paid account to function correctly.

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

## Environment

- Download and install conda if you don’t have it already.
- Create a conda environment based on the yaml found in envs.
  `conda env create --file=envs/build.yaml`

## Tests

- Tests for machine learning can be found in `tests/test_model.py`.
- Tests for FastAPI implementation can be found in `test_api.py`.
- Both of these can be run with `pytest -v` in the command line.

## Continuous Integration

- This is handled by GitHub Actions, the workflow being in the project under `.github`
- The action uses `flake8` to lint and runs the test scripts detailed in the Tests section above.

## Continuous Delivery

- CD is handled by Heroku and is triggered by pushes to the main branch.

## API Demo

- In the `tests` folder, you can find the `api_demo.py` file, which can be executed on the command line to demonstrate the api in action.


## GitHub Actions

- Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
  - Make sure you set up the GitHub Action to have the same version of Python as you used in development.
- Add your <a href="https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions" target="_blank">AWS credentials to the Action</a>.
- Set up <a href="https://github.com/iterative/setup-dvc" target="_blank">DVC in the action</a> and specify a command to `dvc pull`.

## Data

- Download census.csv from the data folder in the starter repository.
  - Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.
- Create a remote DVC remote pointing to your S3 bucket and commit the data.
- This data is messy, try to open it in pandas and see what you get.
- To clean it, use your favorite text editor to remove all spaces.
- Commit this modified data to DVC under a new name (we often want to keep the raw data untouched but then can keep updating the cooked version).

## Model

- Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
- Write unit tests for at least 3 functions in the model code.
- Write a function that outputs the performance of the model on slices of the data.
  - Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
- Write a model card using the provided template.

## API Creation

- Create a RESTful API using FastAPI this must implement:
  - GET on the root giving a welcome message.
  - POST that does model inference.
  - Type hinting must be used.
  - Use a Pydantic model to ingest the body from POST. This model should contain an example.
  - Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
- Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

- Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
- Create a new app and have it deployed from your GitHub repository.
  - Enable automatic deployments that only deploy if your continuous integration passes.
  - Hint: think about how paths will differ in your local environment vs. on Heroku.
  - Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
- Set up DVC on Heroku using the instructions contained in the starter directory.
- Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
- Write a script that uses the requests module to do one POST on your live API.
