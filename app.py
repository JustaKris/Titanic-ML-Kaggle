import os
import numpy as np
import pandas as pd
import logging
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Route for Home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
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
            # print(pred_df.head())
            # print(pred_df.info())
            
            # Logging data frame information
            app.logger.info(f"Prediction DataFrame: \n{pred_df.head()}")
            
            predict_pipeline = PredictPipeline()
            prediction, probability = predict_pipeline.predict(pred_df)
            
            proba_str = round(probability * 100, 1)
            results = f"Survived with a probability of {proba_str}%" if prediction == 1 else f"Did not survive with a probability of {proba_str}%"
            return render_template('home.html', results=results)
        
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return render_template('home.html', results="An error occurred during prediction. Please try again.")

if __name__ == "__main__":
    # Run the application
    app.run(
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 5000)), 
        debug=os.environ.get("DEBUG", "false").lower() == "true"
        )
