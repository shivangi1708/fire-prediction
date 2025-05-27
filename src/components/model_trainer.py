import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "CatBoost": CatBoostClassifier(verbose=False),
                "KNN": KNeighborsClassifier(),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10]
                },
                "Random Forest": {
                    "n_estimators": [64, 128, 256],
                    "max_depth": [None, 10, 20],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [5, 10, 15],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128],
                },
                "XGBoost": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128],
                },
                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7]
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.5, 1.0],
                    "n_estimators": [64, 128]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with acceptable performance.")

            logging.info(f"Best model found: {best_model_name} with score {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred)

            return acc_score

        except Exception as e:
            raise CustomException(e, sys)
