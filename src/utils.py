import os
import sys
import numpy as np
import pandas as pd
import dill

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, object):
    '''
        this function saves our file in pkl format
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(object, file)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        '''
            this function perform model training and return r2_score of all models
        '''
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_r2_score
        return report

    except Exception as e:
        raise CustomException(e, sys)