import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.util import save_object
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def initiate_training(self, file_path):
        try:
            logging.info("Reading dataset")
            df = pd.read_csv(file_path)

            logging.info("Initial shape: {}".format(df.shape))

           
            df["quality"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)

            logging.info("Splitting input and target")
            X = df.drop(columns=["quality"], axis=1)
            y = df["quality"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

           
            save_object(
                file_path=os.path.join("artifacts", "preprocessor.pkl"),
                obj=scaler
            )

           
            train_arr = np.c_[X_train_scaled, y_train]
            test_arr = np.c_[X_test_scaled, y_test]

            logging.info("Starting model training")

            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training completed. Accuracy: {score}")

            return score

        except Exception as e:
            raise CustomException(e, sys)