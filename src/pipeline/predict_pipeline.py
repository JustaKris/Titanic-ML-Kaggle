import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        """Initialize the PredictPipeline class."""
        pass

    def predict(self, features: pd.DataFrame):
        """
        Make a prediction based on the input features.

        Args:
            features (pd.DataFrame): DataFrame containing the input features.

        Returns:
            tuple: A tuple containing the prediction and the probability.
        """
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            model = load_object(model_path)

            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            preprocessor = load_object(preprocessor_path)

            scaled_data = preprocessor.transform(features)

            prediction = model.predict(scaled_data)
            probability = max(model.predict_proba(scaled_data).tolist()[0])

            return prediction, probability
        
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self, age: int,  sex: str, sibsp: int, name_title: str, pclass: str, embarked: str,  cabin_multiple: int) -> None:
        self.age = age
        self.sex = sex
        self.sibsp = sibsp
        self.name_title = name_title
        self.pclass = pclass
        self.embarked = embarked
        self.cabin_multiple = cabin_multiple
        
    def get_data_as_data_frame(self):
        """
        Convert the custom data into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the custom data.
        """

        filler_dict = self._get_fillers()

        try:
            custom_data_input_dict = {
                "Age": [float(self.age)],
                "Sex": [self.sex],
                "SibSp": [int(self.sibsp)],
                "name_title": [self.name_title],
                "Pclass": [int(self.pclass)],
                "Embarked": [self.embarked],
                "cabin_multiple": [int(self.cabin_multiple)],
                "Parch": filler_dict["Parch"],
                # "cabin_letters": filler_dict["cabin_letters"],
                # "numeric_ticket": filler_dict["numeric_ticket"],
                "norm_fare": filler_dict["norm_fare"]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    def _get_fillers(self):
        """
        Returns modes of all columns to serve as stand-ins for some of the non user-friendly features.
        The function tries to approximate some of the features based on the available data.

        Returns:
            dict: A dictionary of mode values for the columns.
        """
        try:
            # Load and merge all available data
            train_df = pd.read_csv(os.path.join("artifacts", "train_augmented.csv"))
            test_df = pd.read_csv(os.path.join("artifacts", "test_augmented.csv"))
            all_data = pd.concat([train_df, test_df])
            
            # Baseline dict of modes to be used as filler for the non user friendly columns
            data_filler_dict = all_data.mode().iloc[0].to_dict()

            # Setting up a filtered df to store all passangers within a given range approximating the provided data
            filtered_data = all_data[(all_data['Sex'] == self.sex) 
                                    #  & (all_data['name_title'] == self.name_title) 
                                    & (all_data['Pclass'] == self.pclass)
                                    & (all_data['Embarked'] == self.embarked)
                                    & (all_data['Age'].between(float(self.age) - 3, float(self.age) + 3))
                                    ]
            print(filtered_data.head())
            # print(filtered_data.info())

            # Replace generic filler odes with case specific ones
            if not filtered_data.empty:
                data_filler_dict["Parch"] = filtered_data["Parch"].mode().iloc[0]
                # data_filler_dict["cabin_letters"] = filtered_data["cabin_letters"].mode().iloc[0]
                # data_filler_dict["numeric_ticket"] = filtered_data["numeric_ticket"].mode().iloc[0]
                data_filler_dict["norm_fare"] = filtered_data["norm_fare"].mean()

            return data_filler_dict
        
        except Exception as e:
            raise CustomException(e, sys)


        
if __name__ == "__main__":
    custom_data = CustomData(
        age=42, 
        sex="female", 
        name_title="Mrs", 
        sibsp=0, 
        pclass=1, 
        embarked="C", 
        cabin_multiple=1
        )
    df = custom_data.get_data_as_data_frame()
    print(df)

    pipeline = PredictPipeline()
    prediction, probability = pipeline.predict(df)
    print(f"Survived with a probability of {round(probability * 100, 1)}%" if prediction == 1 else f"Did not survive with a probability of {round(probability * 100, 1)}%")