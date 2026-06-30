import os
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.logger import logging
from src.exception import CustomException
from src.util import save_object , evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")


class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_arr , test_arr):
        logging.info("split training and testing data")

        try:
            X_train , y_train = train_arr[: , :-1] , train_arr[: , -1]
            X_test , y_test = test_arr[: , :-1] , test_arr[: , -1]

            # 🔥 FIX 1: Convert target (VERY IMPORTANT)
            y_train = np.where(y_train >= 6, 1, 0)
            y_test = np.where(y_test >= 6, 1, 0)

            # 🔥 FIX 2: Scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVC": SVC(),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            params = {
                "Decision Tree": {
                    'criterion':['gini', 'entropy']
                },
                "Random Forest":{
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10]
                },
                "Gradient Boosting":{
                    'learning_rate':[.01,.1],
                    'n_estimators': [100, 200]
                },
                "XGBClassifier":{
                    'learning_rate':[.01,.1],
                    'n_estimators': [100,200]
                }
            }

           
            model_report:dict = evaluate_models(
                X_train= X_train ,
                y_train = y_train ,
                X_test = X_test ,
                y_test = y_test ,
                models = models ,
                param = params
            )

            print("Model Report:", model_report)

            # 🔥 Get best model
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # 🔥 IMPORTANT: Train best model again
            best_model.fit(X_train, y_train)

            print("Best Model:", best_model_name)
            print("Best Score:", best_model_score)

            # 🔥 Save model
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            score = accuracy_score(y_test , predicted)

            return score

        except Exception as e:
            raise CustomException(e , sys)