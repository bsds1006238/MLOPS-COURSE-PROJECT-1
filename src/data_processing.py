
import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from feast import FeatureStore
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataProcessor:
    
     def __init__(self, train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config_path = config_path
        self.config = read_yaml(config_path)
        self.train_test_ratio = self.config["data_ingestion"]["train_ratio"]
        self.store = FeatureStore(repo_path=self.config["data_ingestion"]["feast_repo_path"])
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

     def build_training_data(self):
        try:
            logger.info("Building training data from Feast")
            
            entity_df_query = f"""
            SELECT
              iris_id,
              TIMESTAMP(event_timestamp) AS event_timestamp,
              species
            FROM `{self.config['data_ingestion']['project_id']}.{self.config['data_ingestion']['bq_dataset']}.{self.config['data_ingestion']['bq_table']}`
            """

            training_df = self.store.get_historical_features(
                entity_df=entity_df_query,
                features=[
                    "iris_data:sepal_length",
                    "iris_data:sepal_width",
                    "iris_data:petal_length",
                    "iris_data:petal_width",
                ],
            ).to_df()

            train_df, test_df = train_test_split(
                training_df,
                test_size=1 - self.train_test_ratio,
                random_state=42,
            )

            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)

            logger.info("Training and test data generated via Feast")

        except Exception as e:
            logger.error(f"Failed to build training data with Feast{e}")
            raise CustomException("Failed to build training data with Feast", e)
            
     def process(self):
        try:
            logger.info("Loading data from RAW directory")
            self.build_training_data()
            logger.info("Data processing completed successfully")
            
        except Exception as e:
                logger.error(f"Error in preprocessing pipeline {e}")
                raise CustomException(f"Error while preprocessing pipeline ", e)
            


if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    processor.process()