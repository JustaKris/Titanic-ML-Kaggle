import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns: List[str], categorical_columns: List[str]):
        """
        Creates a ColumnTransformer object for data preprocessing.

        Returns:
            ColumnTransformer: A preprocessing object for transforming numerical and categorical columns.
        """
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Transforms the train and test data.

        Args:
            train_path (str): Path to the train data CSV file.
            test_path (str): Path to the test data CSV file.

        Returns:
            tuple: Transformed train and test data arrays and path to the preprocessor object file.
        """
        try:
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train file not found at path: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file not found at path: {test_path}")

            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            # Base column lists
            numerical_columns = ["PassengerId", "Age", "SibSp", "Parch", "Fare"]
            categorical_columns = ["Name", "Pclass", "Sex", "Ticket", "Cabin", "Embarked"]
            target_column_name = "Survived"

            # Custom preprocessing
            for df in [train_df, test_df]:
                # Feature engineering
                df['cabin_multiple'] = df.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
                df['cabin_adv'] = df.Cabin.apply(lambda x: str(x)[0])
                df['numeric_ticket'] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
                df['ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 0)
                df['name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

                # Log norm of fare
                df['norm_fare'] = np.log(df.Fare+1)

                # Dropping passagers which have not embarked as those records are of no real use
                df.dropna(subset=['Embarked'], inplace=True)

                # Dropping unwanted columns
                columns_to_drop = [
                    "PassengerId",  # Just a unique identifier for each record
                    "Name",  # Person's name will have no relevance
                    "Fare",  # Replaced by norm_fare
                    "Cabin",  # Split into several other features
                    "Ticket",  #
                    "ticket_letters"  # I decided it's not of much relevance
                    ]
                df.drop(columns=columns_to_drop, inplace=True)

            # Additional column lists
            additional_numerical_columns = ["cabin_multiple", "numeric_ticket", "norm_fare"]
            additional_categorical_columns = ["cabin_adv", "ticket_letters", "name_title"]

            # Final feature lists
            final_numerical_columns = [column for column in numerical_columns + additional_numerical_columns if column not in columns_to_drop]
            final_categorical_columns = [column for column in categorical_columns + additional_categorical_columns if column not in columns_to_drop]
            logging.info("Feature engineering done")
        
            # FLoad preprocessor object
            preprocessing_obj = self.get_data_transformer_object(final_numerical_columns, final_categorical_columns)
            logging.info("Obtained preprocessing object")

            # Split target feature
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Fit data to transformer
            train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # For regression tasks
            """
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            """

            logging.info("Data has been fitted to transformer object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Saved preprocessing object.")

            return train_arr, target_feature_train_df.values, test_arr, target_feature_test_df.values, self.data_transformation_config.preprocessor_obj_file_path


        except FileNotFoundError as fnf_error:
            logging.error(fnf_error)
            raise CustomException(fnf_error, sys)
        except pd.errors.EmptyDataError as ede_error:
            logging.error(ede_error)
            raise CustomException(ede_error, sys)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise CustomException(e, sys)

# Testing
if __name__ == "__main__":
    data_transformer = DataTransformation()
    data_ingestor = DataIngestion()

    train_data_path, test_data_path = data_ingestor.initiate_data_ingestion()
    X_train, y_train, X_test, y_test, preprocessor_path = data_transformer.initiate_data_transformation(train_data_path, test_data_path)

    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    print(preprocessor_path)
