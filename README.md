# Titanic Survival Prediction

## Jupyter Notebook

There are two main parts to this project. First is the exploration of the Kaggle Titanic dataset with the aim of tuning a model and submiting predictions to the <b>Titanic - Machine Learning from Disaster</b> competition. All research and model training was done in Jupyter Notebook.

Link to Notebook -> [Titanic-Machine-Learning-from-Disaster](./notebook/Titanic-Machine-Learning-from-Disaster.ipynb)

### Data Overview

The goal is to predict the `Survival` variable (Classification).

There are 11 independent variables (including `PassengerId`):

* `PassengerId` : Unique identifier of each passenger
* `pclass` : Ticket class refering to 1 - 1st, 2 - 2nd, 3 - 3rd
* `sex` : Passenger gender
* `Age` : Passenger age
* `sibsp` : Signifies the number of the passenger's siblings / spouses aboard the Titanic
* `parch` : Signifies the number of the passenger's parents / children aboard the Titanic
* `ticket` : Passenger's ticket number
* `fare` : Ticket cost
* `cabin` : Cabin number
* `embarked` : Port name from which the passenger embarked from

Target variable:
* `Survived`: Boolean value => 0/1 - No/Yes

Kaggle Dataset Link -> [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)

## Predictor Web App

The second part is building an app using what I've learned from the conducted research. The app is designed to be scalable which is achieved by the use of data, model training and prediction pipelines. I have tried to follow python convention so that the app can be deployed to any remote environment.

### Render Deployment

The predictor app was deployed on render using a GitHub Workflow. It's using a basic Rended plan so it might take up to a minute for the app to load if it hasn't been used recently.

Render link -> [https://titanic-ml-kaggle.onrender.com](https://titanic-ml-kaggle.onrender.com)

### Web App Approach

1. Data Ingestion: 
    * The Data Ingestion script reads the data in the format provided in Kaggle which is separate CSV files for the training and test sets. 
    * Since the provided test data file does not contain any labels, the training data get split and saved into separate train and test data CSVs.

2. Data Transformation: 
    * In this phase a ColumnTransformer Pipeline is created to handle all of the data transfromation.
    * SimpleImputer is used for Numeric features with a strategy of `median`. Then the data is scaled using StandardScaler. The only exception is `Fare`which gets replaced by `norm_fare` a normalized (log(Fare + 1)) version of the feature.
    * SimpleImputer is used for Categorical Features as well but with a strategy of `most frequent`. OneHotEncoder and StandardScaler are applied next.
    * This preprocessor is saved to a pickle file for later use.
    * Feature engineering - a few features were created in order to either simplify an existing feature or try and derive more detailed information from it:
        - `cabin_multiple`: Derived from the `cabin` feature with the aim of figuring out if the number of passengers per cabin has any relevance
        - `name_title`: Pulled from passenger names and serves a very similar purpose to gender
    

3. Model Training: 
    * This script trains and evaluates a list of provoded models.
    * Hyperparameters for the most promissing models are tuned as well.
    * A VotingCalssifier is used at the end since that seemed to provide the best results. It utilises the best performing tuned models.
    * The best performing model is saved for later use.

4. Prediction Pipeline: 
    * This pipeline utilizes the saved model and preprocessor object, loading each from the respective pickle file in order to predict a given datapoint and return the prediction along with the model's level of certainty.

5. Flask App:
    * A simple Flask app houses the user interface where input is received and passed to the prediction pipeline. The app then displays the resulting outcome.