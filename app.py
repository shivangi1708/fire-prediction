
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

def time_of_day_to_num(tod):
    mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    return mapping.get(tod.lower(),0)


application=Flask(__name__)

app=application

## Route for a home page

'''@app.route('/')
def index():
    return render_template('index.html') '''

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            collector_id = float(request.form.get('collector_id', 0)),
            month = float(request.form.get('month', 0)),
            time_of_day = time_of_day_to_num(request.form.get('time_of_day', 'morning')),
            temperature = float(request.form.get('temperature', 0)),
            humidity = float(request.form.get('humidity', 0)),
            wind_intensity = float(request.form.get('wind_intensity', 0)),
            rain = float(request.form.get('rain', 0)),
            surface_litter = float(request.form.get('surface_litter', 0)),
            tree_age = float(request.form.get('tree_age', 0)),
            tree_density = float(request.form.get('tree_density', 0)),
            l_score = float(request.form.get('l_score', 0)),
            c_score = float(request.form.get('c_score', 0)),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    
    

if __name__=="__main__":
    app.run(host="0.0.0.0")

