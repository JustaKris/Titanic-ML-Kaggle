import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            model = load_object(model_path)

            preprocessor_path = 'artifacts\preprocessor.pkl'
            preprocessor = load_object(preprocessor_path)

            scaled_data = preprocessor.transform(features)

            prediction = model.predict(scaled_data)
            probability = max(model.predict_proba(scaled_data).tolist()[0])

            return prediction, probability
        
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(
            self, 
            age: int, 
            sex: str, 
            sibsp: int, 
            name_title: str, 
            pclass: str, 
            # parch: int, 
            embarked: str, 
            # cabin_adv: str, 
            cabin_multiple: int, 
            # numeric_ticket: int, 
            # norm_fare: float
            ) -> None:
        self.age = age
        self.sex = sex
        self.sibsp = sibsp
        self.name_title = name_title
        self.pclass = pclass
        # self.parch = parch
        self.embarked = embarked
        # self.cabin_adv = cabin_adv
        self.cabin_multiple = cabin_multiple
        # self.numeric_ticket = numeric_ticket
        # self.norm_fare = norm_fare
        
    def get_data_as_data_frame(self):

        mode_dict = self._get_fillers()

        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Sex": [self.sex],
                "SibSp": [self.sibsp],
                "name_title": [self.name_title],
                "Pclass": [self.pclass],
                "Parch": mode_dict["Parch"],
                "Embarked": [self.embarked],
                "cabin_adv": mode_dict["cabin_adv"],
                "cabin_multiple": [self.cabin_multiple],
                "numeric_ticket": mode_dict["numeric_ticket"],
                "norm_fare": mode_dict["norm_fare"]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    def _get_fillers(self):
        '''
        Returns modes of all columns to serve as standins for some of the non user friendly features.
        The functions tries to approximate some of the features based on the available data.
        '''
        # Load and merge all available data
        train_df = pd.read_csv("artifacts/train_augmented.csv")
        test_df = pd.read_csv("artifacts/test_augmented.csv")
        all_data = pd.concat([train_df, test_df])
        
        # Baseline dict of modes to be used as filler for the non user friendly columns
        data_filler_dict = all_data.mode().to_dict(orient='records')[0]

        # Setting up a filtered df to store all passangers within a given range approximating the provided data
        filtered_data = all_data[(all_data['Sex'] == self.sex) 
                                #  & (all_data['name_title'] == self.name_title) 
                                 & (all_data['Pclass'] == self.pclass)
                                 & (all_data['Embarked'] == self.embarked)
                                 & (all_data['Age'] <= float(self.age + 3))
                                 & (all_data['Age'] >= (float(self.age - 3)))
                                 ]
        # print(filtered_data.head())

        # Replace generic filler odes with case specific ones
        if not filtered_data.empty:
            data_filler_dict["Parch"] = filtered_data["Parch"].mode()
            data_filler_dict["cabin_adv"] = filtered_data["cabin_adv"].mode()
            data_filler_dict["numeric_ticket"] = filtered_data["numeric_ticket"].mode()
            data_filler_dict["norm_fare"] = filtered_data["norm_fare"].mean()

        return data_filler_dict


        
if __name__ == "__main__":
    # custom_data = CustomData(25, "female", 1, "Mrs.", 1, 1, "S", "C", 1, 0, 3.22)
    custom_data = CustomData(age=42, sex="female", sibsp=0, name_title="Mrs", pclass=1, embarked="C", cabin_multiple=1)
    df = custom_data.get_data_as_data_frame()
    print(df)

    pipeline = PredictPipeline()
    prediction, probability = pipeline.predict(df)
    print(f"Survived with a probability of {round(probability * 100, 1)}%" if prediction == 1 else f"Did not survive with a probability of {round(probability * 100, 1)}%")