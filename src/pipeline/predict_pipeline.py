import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Loading model and preprocessor...")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            print("Transforming input features...")
            transformed_data = preprocessor.transform(features)

            print("Making predictions...")
            predictions = model.predict(transformed_data)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        collector_id: float,
        month: float,
        time_of_day: str,
        temperature: float,
        humidity: float,
        wind_intensity: float,
        rain: float,
        surface_litter: float,
        tree_age: float,
        tree_density: float,
        l_score: float,
        c_score: float,
    ):
        self.collector_id = collector_id
        self.month = month
        self.time_of_day = time_of_day
        self.temperature = temperature
        self.humidity = humidity
        self.wind_intensity = wind_intensity
        self.rain = rain
        self.surface_litter = surface_litter
        self.tree_age = tree_age
        self.tree_density = tree_density
        self.l_score = l_score
        self.c_score = c_score

    def get_data_as_data_frame(self):
        try:
            input_dict = {
                "collector.id": [self.collector_id],
                "month": [self.month],
                "time.of.day": [self.time_of_day],
                "temperature": [self.temperature],
                "humidity": [self.humidity],
                "wind.intensity": [self.wind_intensity],
                "rain": [self.rain],
                "surface.litter": [self.surface_litter],
                "tree.age": [self.tree_age],
                "tree.density": [self.tree_density],
                "l.score": [self.l_score],
                "c.score": [self.c_score],
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)

