import pandas as pd
from unittest import TestCase, main
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


class TestCustomData(TestCase):
    def test_get_data_as_data_frame(self):
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
        expected_columns = ["Age", "Sex", "SibSp", "name_title", "Pclass", "Embarked", "cabin_multiple", "Parch", "norm_fare"]
        self.assertListEqual(list(df.columns), expected_columns)
        self.assertListEqual(df.Age.tolist(), [float(custom_data.age)])


class TestPredictPipeline(TestCase):
    def test_predict_success(self):
        
        features = pd.DataFrame({
            "Age": [42],
            "Sex": ["female"],
            "SibSp": [0],
            "name_title": ["Mrs"],
            "Pclass": [1],
            "Embarked": ["C"],
            "cabin_multiple": [1],
            "Parch": [0],
            "norm_fare": [4.2]
        })
        
        pipeline = PredictPipeline()
        prediction, probability = pipeline.predict(features)
        
        self.assertEqual(prediction, 1)
        self.assertAlmostEqual(probability, 0.9775034151903974)


if __name__ == '__main__':
    main()
