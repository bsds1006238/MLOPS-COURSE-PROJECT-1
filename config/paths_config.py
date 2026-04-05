import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

################### DATA INGESTION #################

RAW_DIR = BASE_DIR / "artifacts" / "raw"
RAW_FILE_PATH = os.path.join(RAW_DIR,"iris.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR,"train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR,"test.csv")

CONFIG_PATH = 'config/config.yaml'


################ Data Processing ##########


PROCESSED_DIR = BASE_DIR / 'artifacts' / 'processed'
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR,"processed_test.csv")


################ MODEL OUTPUT PATH ##########

MODEL_OUTPUT_PATH = BASE_DIR / 'artifacts' / 'model' / 'rf_model.pkl'