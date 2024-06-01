import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    # Default data input path
    raw_train_data_path: str = os.path.join('notebook', 'data', 'train.csv')
    raw_test_data_path: str = os.path.join('notebook', 'data', 'test.csv')
    # Export paths for safekeeping
    processed_train_data_path: str = os.path.join('artifacts', 'train_processed.csv')
    processed_test_data_path: str = os.path.join('artifacts', 'test_processed.csv')

class DataIngestion:
    def __init__(self, raw_train_data_path: str = None, raw_test_data_path: str = None):
        """
        Initialize the DataIngestion class.

        Args:
            raw_train_data_path (str): Path to the raw train data CSV file. Defaults to the configured path if not provided.
            raw_test_data_path (str): Path to the raw test data CSV file. Defaults to the configured path if not provided.
        """
        self.ingestion_config = DataIngestionConfig()
        if raw_train_data_path:
            self.ingestion_config.raw_train_data_path = raw_train_data_path
        # if raw_test_data_path:
        #     self.ingestion_config.raw_test_data_path = raw_test_data_path
    
    def initiate_data_ingestion(self):
        """
        Reads the raw train and test data, and saves the datasets with 'output' in the filenames.

        Returns:
            tuple: Paths to the processed train and test data files.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the raw train and test data
            df = pd.read_csv(self.ingestion_config.raw_train_data_path)
            # train_df = pd.read_csv(self.ingestion_config.raw_train_data_path)
            # test_df = pd.read_csv(self.ingestion_config.raw_test_data_path)
            logging.info(f"Read the dataset as a dataframe with shape: {df.shape}")
            # logging.info(f"Read the train dataset as a dataframe with shape: {train_df.shape}")
            # logging.info(f"Read the test dataset as a dataframe with shape: {test_df.shape}")

            # Ensuring the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.processed_train_data_path), exist_ok=True)

            # Train Test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Saving the processed train and test data with 'output' in the filenames
            train_df.to_csv(self.ingestion_config.processed_train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.processed_test_data_path, index=False, header=True)
            logging.info(f"Processed train data saved at: {self.ingestion_config.processed_train_data_path}")
            logging.info(f"Processed test data saved at: {self.ingestion_config.processed_test_data_path}")

            return self.ingestion_config.processed_train_data_path, self.ingestion_config.processed_test_data_path
        
        except FileNotFoundError as fnf_error:
            logging.error("File not found: %s", fnf_error)
            raise CustomException(fnf_error, sys)
        except pd.errors.EmptyDataError as ede_error:
            logging.error("Empty data error: %s", ede_error)
            raise CustomException(ede_error, sys)
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise CustomException(e, sys)

# Testing
if __name__ == "__main__":
    obj = DataIngestion()
    processed_train_data, processed_test_data = obj.initiate_data_ingestion()
    print(processed_train_data, processed_test_data)
