import os
import sys
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
from src.components.model_trainer import ModelTrainer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _create_column_transformer(self, numerical_columns: List[str], categorical_columns: List[str]):
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

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # Feature engineering
        df['cabin_multiple'] = df.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
        # df['cabin_letters'] = df.Cabin.apply(lambda x: str(x)[0])
        # df['numeric_ticket'] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
        # df['ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() if len(x.split(' ')[:-1]) > 0 else 0)
        df['name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

        # # Log norm of fare
        df['norm_fare'] = np.log(df.Fare + 1)

        # Dropping passagers which have not embarked as those records are of no real use
        df.dropna(subset=['Embarked'], inplace=True)

        # Dropping unwanted columns
        columns_to_drop = [
            "PassengerId",  # Just a unique identifier for each record
            "Name",  # Person's name will have no relevance
            "Fare",  # Replaced by norm_fare
            "Cabin",  # Split into several other features
            "Ticket"  # Just ticket ids
        ]
        df.drop(columns=columns_to_drop, inplace=True)

        return df

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train file not found at path: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file not found at path: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            train_df = self._apply_feature_engineering(train_df)
            test_df = self._apply_feature_engineering(test_df)

            # Saving mode values of each column for later use
            train_df.to_csv(os.path.join('artifacts', 'train_augmented.csv'), index=False, header=True)
            test_df.to_csv(os.path.join('artifacts', 'test_augmented.csv'), index=False, header=True)

            # Column lists for each preprocessor pipeline
            # numerical_columns = ["Age", "SibSp", "Parch", "norm_fare", "cabin_multiple", "numeric_ticket"]
            # categorical_columns = ["Pclass", "Sex", "Embarked", "cabin_letters", "name_title"]
            numerical_columns = ["Age", "SibSp", "Parch", "norm_fare", "cabin_multiple"]
            categorical_columns = ["Pclass", "Sex", "Embarked", "name_title"]

            # Preprocessing
            preprocessor = self._create_column_transformer(numerical_columns, categorical_columns)

            X_train = preprocessor.fit_transform(train_df.drop(columns=["Survived"]))
            y_train = train_df["Survived"].values

            X_test = preprocessor.transform(test_df.drop(columns=["Survived"]))
            y_test = test_df["Survived"].values

            logging.info("Preprocessed dataframes.")
            
            # Saving preprocessor object for future use
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            return X_train, y_train, X_test, y_test, self.data_transformation_config.preprocessor_obj_file_path

        except FileNotFoundError as fnf_error:
            logging.error(f"File not found: {fnf_error}")
            raise CustomException(fnf_error, sys)
        except pd.errors.EmptyDataError as ede_error:
            logging.error(f"Empty Data Error: {ede_error}")
            raise CustomException(ede_error, sys)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_transformer = DataTransformation()
    data_ingestor = DataIngestion()
    model_trainer = ModelTrainer()

    train_data_path, test_data_path = data_ingestor.initiate_data_ingestion()
    X_train, y_train, X_test, y_test, preprocessor_path = data_transformer.initiate_data_transformation(train_data_path, test_data_path)

    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)
    # print(preprocessor_path)

    best_score = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    print(best_score)
