import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
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
        logging.info('Entered into the data Ingestion method')
        try:
            data = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the data as dataframe')

            data['target'] = round((data['math_score'] + data['reading_score'] + data['writing_score'])/3, 3)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test Initiated')
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.raw_data_path

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data, _ = obj.initiate_data_ingestion()
    data_trans = DataTransformation()
    data_trans.initiate_data_transformation(train_data, test_data)
            