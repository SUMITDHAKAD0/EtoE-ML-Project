import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig 
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path :str= os.path.join('artifacts', 'train.csv')
    test_data_path :str= os.path.join('artifacts', 'test.csv')
    raw_data_path :str= os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Starting Data Ingestion')
        try:
            # reading data from directry 
            data = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the data as dataframe')

            # creating target column( taking avg. of all marks) 
            # data['target'] = round((data['math_score'] + data['reading_score'] + data['writing_score'])/3, 3)

            # saving data to the directary
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test Initiated')
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ending Data Ingestion')

            #returning data path
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.raw_data_path

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data, _ = obj.initiate_data_ingestion()

    data_trans = DataTransformation()
    train_arr,test_arr, _ =  data_trans.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2_score, model_name = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print('Model = ',model_name,' ', 'r2_score = ',r2_score)