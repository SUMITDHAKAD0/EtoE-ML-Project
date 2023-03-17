import os
import sys
import json
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    trained_model_report = os.path.join("artifacts", "report_model.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info('Starting Model Training')

            #Train test split
            x_train, y_train, x_test, y_test = (
                    train_data[:,:-1],
                    train_data[:,-1],
                    test_data[:,:-1],
                    test_data[:,-1]
                )
            
            # Initialize models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Initialize hyperparameters for all models
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Classifier":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }     
            }

            # calling model training function 
            model_report:dict = evaluate_models(
                x_train = x_train, 
                y_train = y_train, 
                x_test = x_test, 
                y_test = y_test, 
                models = models,
                params = params
            )

            #saving model report
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_report), exist_ok=True)
            report_df = pd.DataFrame(model_report, index=list(model_report.keys()))
            report_df.to_csv(self.model_trainer_config.trained_model_report) 

            # Finding best r2 score
            best_model_r2score = max(sorted(model_report.values()))

            # Finding best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_r2score)] 
        
            best_model = models[best_model_name]

            if best_model_r2score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")

            # saving best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object = best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)

            logging.info("Ending Model Training")

            return r2_square, best_model_name

        except Exception as e:
            raise CustomException(e, sys)