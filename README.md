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
- Swagger docs for the app can be found [here](https://desert-planet.herokuapp.com/docs).
