from src.pipeline.predict_pipeline import CustomData, PredictPipeline

input_data = CustomData(2,8,"night",28.5,55.0,3.0,0.0,6.0,20.0,50.0,0.68,0.45)

df = input_data.get_data_as_data_frame()
pipeline = PredictPipeline()
prediction = pipeline.predict(df)

print(f"🔥 Fire Risk Prediction: {prediction[0]}")

