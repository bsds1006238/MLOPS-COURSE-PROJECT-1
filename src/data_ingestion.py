import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
from google.cloud import storage
from google.cloud import bigquery

logger = get_logger(__name__)




def build_bq_schema(schema_config: list):
    return [
        bigquery.SchemaField(col["name"], col["type"])
        for col in schema_config
    ]
    
    
    
class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.project_id = self.config["project_id"]
        self.train_test_ratio = self.config["train_ratio"]
        os.makedirs(RAW_DIR, exist_ok=True)
        self.dataset_id = self.config["bq_dataset"]
        self.table_id = self.config["bq_table"]

        os.makedirs(RAW_DIR, exist_ok=True)
        self.bq_client = bigquery.Client(project=self.project_id)
        logger.info("Data Ingestion initialized")
    
    def upload_to_bigquery(self):
        try:
            df = pd.read_csv(RAW_FILE_PATH)
            df.drop(columns=["iris_id","event_timestamp"], inplace=True)
            df["iris_id"] = "GLOBAL"
            df["event_timestamp"] = pd.to_datetime("2026-04-01 00:00:00")
            df.to_csv(RAW_FILE_PATH, index=False)
            
            logger.info("Data prepared for Feast")
            
            client = bigquery.Client()
            dataset_id = f"{client.project}.{self.dataset_id}"

            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"  # or your region
            client.create_dataset(dataset, exists_ok=True)

            
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            
            schema = build_bq_schema(self.config["schema"])

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                schema=schema,
                skip_leading_rows=1,
                write_disposition="WRITE_TRUNCATE",
            )

            with open(RAW_FILE_PATH, "rb") as f:
                load_job = self.bq_client.load_table_from_file(
                    f, table_ref, job_config=job_config
                )

            load_job.result()
            logger.info(f"Uploaded data to BigQuery table {table_ref}")

        except Exception as e:
            logger.error("Error in uploading data to BigQuery",e)
            raise CustomException("Failed to upload data to BigQuery", e)
        
    def split_data(self):
        try:
            logger.info("Starting the splitting data")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data,test_size = 1-self.train_test_ratio,random_state=42)
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)
            logger.info(f"TRAIN data saved to {TRAIN_FILE_PATH}")
            logger.info(f"TEST data saved to {TEST_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error in splitting data",e)
            raise CustomException(f"Failed to split data into training and test sets", e)
    
    def run(self):
        try:
            logger.info("Starting Data ingestion process")
            self.upload_to_bigquery()
            self.split_data()
            logger.info("Data ingestion completed succesffuly")
            
        except CustomException as ce:
            logger.error(f"Custom Exception : {str(ce)}")
            
        finally:
            logger.info("Data ingestion completed")
            

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
    