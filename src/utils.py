import os
import sys
import dill
import logging
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
