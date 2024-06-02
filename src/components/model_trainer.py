import os
import sys
import itertools
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, optimise_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            # Dictionary of model to evaluate
            models = {
                'Logistic Regression': LogisticRegression(max_iter=2000),
                # 'Random Forest Classifier': RandomForestClassifier(random_state=1),
                # 'KNeighbors Classifier': KNeighborsClassifier(),
                # 'SVC': SVC(probability=True),
                'XGB Classifier': XGBClassifier(random_state=1)
            }

            # Hyperparameter dictionary for each model
            params = {
                'Logistic Regression': {
                    'max_iter' : [2000],
                    'penalty' : ['l1', 'l2'],
                    'C' : np.logspace(-4, 4, 20),
                    'solver' : ['liblinear']
                    },
                'Random Forest Classifier': {
                    'n_estimators': [400, 450, 500, 550],
                    'criterion':['gini', 'entropy'], 
                    'bootstrap': [True],
                    'max_depth': [15, 20, 25],
                    'max_features': ['sqrt', 'log2', 10],
                    'min_samples_leaf': [2,3],
                    'min_samples_split': [2,3]
                    },
                'KNeighbors Classifier': {
                    'n_neighbors' : [3, 5, 7, 9],
                    'weights' : ['uniform', 'distance'],
                    'algorithm' : ['ball_tree', 'kd_tree'],
                    'p' : [1, 2]
                    },
                'SVC': [
                    {'kernel': ['rbf'], 'gamma': [.1, .5, 1, 2, 5, 10], 'C': [.1, 1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'degree' : [2, 3, 4, 5], 'C': [.1, 1, 10, 100, 1000]}
                    ],
                'XGB Classifier': {
                    'n_estimators': [450, 500, 550],
                    'colsample_bytree': [0.75, 0.8, 0.85],
                    'max_depth': [None],
                    'reg_alpha': [1],
                    'reg_lambda': [2, 5, 10],
                    'subsample': [0.55, 0.6, .65],
                    'learning_rate':[0.5],
                    'gamma':[.5, 1, 2],
                    'min_child_weight':[0.01],
                    'sampling_method': ['uniform']
                    },
            }
            
            # Evaluate models
            model_report = optimise_models(
                models=models, 
                params=params, 
                X_train=X_train, 
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
                )
            
            # Voting Classifier fiasco
            tuned_model_tuples = [(model_name, model[0]) for model_name, model in model_report.items()]
            voting_clf = VotingClassifier(estimators=tuned_model_tuples, voting='soft')

            # In a soft voting classifier you can apply weights to each of the models. Let's use a grid search to explore different weightings:
            combinations = itertools.product([1, 2], repeat=len(tuned_model_tuples))  # Generate all possible combinations of weights
            combinations = [list(comb) for comb in combinations if len(set(comb)) != 1]  # Filter out the combinations where all elements are the same
            
            voting_classifier_params = {'Voting Classifier': {'weights': combinations, 'voting': ['soft', 'hard']}}

            tuned_voting_clf = optimise_models(
                {'Voting Classifier': voting_clf},
                voting_classifier_params,
                X_train,
                y_train,
                X_test,
                y_test
            )
            model_report['Voting Classifier'] = tuned_voting_clf['Voting Classifier']
            model_report = dict(sorted(model_report.items(), key=lambda item: -item[1][1]))

            # Retrieve models and return sorted list of tuples by score
            best_model_name, best_model_tuple = list(model_report.items())[0]
            best_model, best_model_score = best_model_tuple
                
            # Check if top model is above a minimum score threshold
            if best_model_score < 0.6:
                raise CustomException("Model score too low")
            
            logging.info("Found best model")

            # Save best performing model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Get best model predictions and retrun r2 score
            # predicted = best_model.predict(X_test)

            # return r2_score(y_test, predicted)
            return best_model_score
        
        except Exception as e:
            raise CustomException(e, sys)