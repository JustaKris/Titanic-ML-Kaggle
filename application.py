from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

# Route for Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=request.form.get('age'), 
            sex=request.form.get('gender').lower(), 
            name_title=request.form.get('name_title'), 
            sibsp=request.form.get('sibsp'), 
            pclass=request.form.get('pclass'), 
            embarked=request.form.get('embarked'), 
            cabin_multiple=request.form.get('cabin_multiple')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df.head())
        print(pred_df.info())

        predict_pipeline = PredictPipeline()
        prediction, probability = predict_pipeline.predict(pred_df)

        results = f"Survived with a probability of {round(probability * 100, 1)}%" if prediction == 1 else f"Did not survive with a probability of {round(probability * 100, 1)}%"

        return render_template('home.html', results=results)
    

if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True)
    app.run(host="0.0.0.0")