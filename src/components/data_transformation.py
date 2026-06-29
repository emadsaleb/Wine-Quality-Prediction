import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.util import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "fixed_acidity",
                "volatile_acidity",
                "citric_acid",
                "residual_sugar",
                "chlorides",
                "free_sulfur_dioxide",
                "total_sulfur_dioxide",
                "density",
                "ph",
                "sulphates",
                "alcohol",
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self , train_path , test_path):
        try:
            train_wine_df = pd.read_csv(train_path)
            test_wine_df = pd.read_csv(test_path)

            # Clean column names
            train_wine_df.columns = train_wine_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
            test_wine_df.columns = test_wine_df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "quality"

            # Split features and target
            input_feature_train = train_wine_df.drop(columns=[target_column_name])
            input_feature_test = test_wine_df.drop(columns=[target_column_name])

            target_feature_train = train_wine_df[target_column_name]
            target_feature_test = test_wine_df[target_column_name]

            logging.info("Applying preprocessing on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)

            train_arr = np.c_[input_feature_train_arr, target_feature_train.values]
            test_arr = np.c_[input_feature_test_arr, target_feature_test.values]

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)