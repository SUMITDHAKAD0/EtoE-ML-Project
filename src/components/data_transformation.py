import os
import sys
import numpy as np
import pandas as pd

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_data_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ['math_score', 'reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            logging.info('Numerical column scalling done')

            cat_pipeline = Pipeline(
                steps = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoding', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=0))
                ]
            )

            logging.info('Categorical column encoding done')
            logging.info(f'Categorical columns : {categorical_features}')
            logging.info(f'Numerical columns : {numerical_features}')

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            logging.info('ColumnTransformation done')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('train and test data loades done')

            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'target'

            x_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]

            x_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            logging.info('Train Test spliting Done')

            x_train_preproccesed = preprocessor_obj.fit_transform(x_train)
            x_test_preproccesed = preprocessor_obj.fit_transform(x_test)

            train_arr = np.c_[x_train_preproccesed, np.array(y_train)]
            test_arr = np.c_[x_test_preproccesed, np.array(y_test)]

            logging.info('Data Transformation done')

            save_object(
                file_path = self.data_transformation_config.preprocessor_data_path,
                object = preprocessor_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)