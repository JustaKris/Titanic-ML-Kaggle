import os
import sys
import dill
import warnings
import logging
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import f1_score, make_scorer
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves a given object to a specified file path.

    Args:
        file_path (str): The path where the object will be saved.
        obj: The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved to {file_path}")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)
    

def load_object(file_path):
    """
    Loads an object from a specified file path.

    Args:
        file_path (str): The path from where the object will be loaded.

    Returns:
        obj: The loaded object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            loaded_obj = dill.load(file_obj)

        logging.info(f"Object loaded from {file_path}")

        return loaded_obj
    
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)
    

def optimise_models(models: dict, params: dict, X_train, y_train, X_test=None, y_test=None) -> dict:
    '''
    This function uses dictionaries of models and parameters on which it will run a grid 
    search cross-validation to determine which option is the optimal one for each model.
    Returns a dictionary of models with their best estimators applied.
    '''
    tuned_models = {}

    print("\n------------ Start of model tuning ------------\n")
    
    for model_name, model in models.items():
        
        print(f"{model_name}:")
        
        f1_scorer = make_scorer(f1_score, average='weighted')  # Use weighted F1 score for multi-class classification with imbalanced class distribution

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
        
            # Setup GridSearchCV for current model
            tuned_model = GridSearchCV(
                model,
                param_grid=params[model_name],
                cv=5,
                scoring=f1_scorer,
                verbose=True,
                n_jobs=-1
            )
            
            # Fit data
            tuned_model.fit(X_train, y_train)

            # Save best models and best scores
            best_tuned_model = tuned_model.best_estimator_
            tuned_models[model_name] = [best_tuned_model, tuned_model.best_score_]

            # Report model scores and best parameters
            print('- Best Parameters: ' + str(tuned_model.best_params_))
            print('- Best F1 Score Train: ' + str(round(tuned_model.best_score_ * 100, 1)) + '%')

            # Calculate F1 score on test set if available
            if X_test is not None and y_test is not None:
                y_pred_test = best_tuned_model.predict(X_test)
                f1_test_score = f1_score(y_test, y_pred_test, average='weighted')
                print('- Best F1 Score Test: ' + str(round(f1_test_score * 100, 1)) + '%')
                tuned_models[model_name][1] = f1_test_score
            print()

    # Return sorted dictionary of model names, models and scores
    sorted_tuned_models = dict(sorted(tuned_models.items(), key=lambda item: -item[1][1]))
    return sorted_tuned_models
