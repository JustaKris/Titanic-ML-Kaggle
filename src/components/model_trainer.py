import os
import sys
import itertools
import warnings
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, optimise_models

warnings.filterwarnings('ignore')


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
                # 'Logistic Regression': LogisticRegression(max_iter=2000),  # performs ok but keeps throwing warnings that I can't seem to get rid of
                'Random Forest Classifier': RandomForestClassifier(random_state=1),
                'KNeighbors Classifier': KNeighborsClassifier(),
                # 'SVC': SVC(probability=True),
                'XGB Classifier': XGBClassifier(random_state=1),
                'CatBoost Classifier': CatBoostClassifier(silent=True)
            }

            # Hyperparameter dictionary for each model
            params = {
                'Logistic Regression': {
                    'max_iter' : [10, 50, 100, 1000],
                    'penalty' : ['l1', 'l2'],
                    'C' : np.logspace(-4, 4, 20),
                    'solver' : ['liblinear']
                    },
                'Random Forest Classifier': {
                    'n_estimators': [100, 300, 450],
                    'criterion':['gini', 'entropy'], 
                    'bootstrap': [True],
                    'max_depth': [10, 15],
                    'max_features': ['sqrt', 20, 40],
                    'min_samples_leaf': [1, 2],
                    'min_samples_split': [.5, 2]
                    },
                'KNeighbors Classifier': {
                    'n_neighbors' : [3, 5, 7, 9, 12, 15],
                    'weights' : ['uniform', 'distance'],
                    'algorithm' : ['ball_tree', 'kd_tree'],
                    'p' : [1, 2]
                    },
                'SVC': [
                    {'kernel': ['rbf'], 'gamma': [.1, .5, 1, 2, 5, 10], 'C': [.1, 1, 10]},
                    {'kernel': ['linear'], 'C': [.1, 1, 10]},
                    {'kernel': ['poly', 'linear', 'rbf'], 'degree' : [1, 2, 3], 'C': [.1, 1, 10]}
                    ],
                'XGB Classifier': {
                    'n_estimators': [500, 550],
                    'colsample_bytree': [.5, .6, .75],
                    'max_depth': [10, None],
                    'reg_alpha': [1],
                    'reg_lambda': [5, 10, 15],
                    'subsample': [.55, .6, .65],
                    'learning_rate':[.5],
                    'gamma':[.25, .5, 1],
                    'min_child_weight':[0.01],
                    'sampling_method': ['uniform']
                    },
                'CatBoost Classifier': {
                    'iterations': [400, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [2, 4, 6],
                    'l2_leaf_reg': [1, 2],
                    'border_count': [64, 128]
                }
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
                raise CustomException("No model with an F1 score above 60%")
            
            logging.info(f"Best model -> {best_model_name} with a F1 score of {round(best_model_score * 100, 1)}%")

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