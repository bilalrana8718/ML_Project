import os 
import sys 
from exec.exception import CustomException
from exec.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig
from model_trainer import ModelTrainer
from model_trainer import ModelConfig


@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', "raw_data.csv")
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def ingest_data(self):
        logging.info("Entered ingesting data")
        try:
            df=pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Read the dataSet as pd')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Splitting data into train and test sets")
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data split completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.ingest_data()

    data_transform = DataTransformation()
    train_arr,test_arr,pre_path = data_transform.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.train_model(train_arr, test_arr, pre_path))


