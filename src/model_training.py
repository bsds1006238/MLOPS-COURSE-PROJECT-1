import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform


import mlflow
import mlflow.sklearn

from scipy.stats import randint

logger = get_logger(__name__)


class ModelTraining:
    
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.parms_dist = MODEL_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=['iris_id','event_timestamp','species'])
            y_train = train_df['species']
            
            
            #iris_id,event_timestamp,species,sepal_length,sepal_width,petal_length,petal_width
            X_test = test_df.drop(columns=['iris_id','event_timestamp','species'])
            y_test = test_df['species']
            
            logger.info(f"Data splitted successfully for Model training")
            
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error (f"Error while loading data {e}")
            raise CustomException ("Failed to load data", e)
        
    def train_rf(self,X_train,y_train):
        try:
            logger.info("Initializing our model")
            
            pipe = Pipeline([
                ("scaler", StandardScaler()),   # will be ignored by RF but OK; or remove if you don't need scaling
                ("clf", RandomForestClassifier(random_state=42))
            ])

            logger.info("Starting our HyperParameter tuning")
            
            random_search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=self.parms_dist,  # <- list of dicts (MODEL_PARAMS)
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"],
                refit=True
            )

            logger.info("Starting our Hyper Parameter training")
            
            random_search.fit(X_train,y_train)
            
            logger.info("HyperParameter Tuning completed")
            
            best_params = random_search.best_params_
            best_model = random_search.best_estimator_
            
            logger.info(f"Best Parameters are : {best_params}")
            
            return random_search
        
        except Exception as e:
            logger.error (f"Error while training model {e}")
            raise CustomException ("Failed to train model", e)
    
    def evaluate_model(self,model,X_test,y_test):
        try:
            logger.info("Evaluating our model")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred,average="macro")
            recall = recall_score(y_test,y_pred,average="macro")
            f1 = f1_score(y_test,y_pred,average="macro")
            
            logger.info(f"Accuracy score : {accuracy}")
            logger.info(f"Precision score : {precision}")
            logger.info(f"Recall score is {recall}")
            logger.info(f"f1 score is {f1}")
            
            return {
                "accuracy" : accuracy,
                "precision" : precision,
                "recall" : recall,
                "f1" : f1
            }
            
        except Exception as e:
            logger.error (f"Error while training model {e}")
            raise CustomException ("Failed to train model", e)
    
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            
            logger.info("Saving the model")
            
            joblib.dump(model,self.model_output_path)
            
            logger.info(f"Model saved to {self.model_output_path}")
        
        except Exception as e:
            logger.error (f"Error while Saving model {e}")
            raise CustomException ("Failed to save model", e)
    
    def run(self):
        try:
            mlflow.set_experiment("OPPE1-Model-Training")
            with mlflow.start_run(run_name="random_search_parent"):
                logger.info("Starting our Model Training Pipeline")
                logger.info("starting our MLFLOW Experimentation")
                logger.info("Logging the training and test dataset to MLFLOW")
                
                mlflow.log_artifact(self.train_path,artifact_path="datasets")
                mlflow.log_artifact(self.test_path,artifact_path="datasets")
                
                
                X_train,y_train,X_test,y_test = self.load_and_split_data()
                random_search = self.train_rf(X_train,y_train)
                
                
                logger.info("Logging all CV trials to MLflow")

                cv_results = random_search.cv_results_


                for i, params in enumerate(cv_results["params"]):
                                model_name = params["clf"].__class__.__name__
                                mean_cv_score = cv_results["mean_test_score"][i]

                                with mlflow.start_run(
                                    run_name=f"trial_{i}_{model_name}",
                                    nested=True
                                ):
                                    mlflow.log_param("model_type", model_name)
                                    mlflow.log_metric("mean_cv_score", mean_cv_score)

                                    # log hyperparameters (exclude estimator object)
                                    clean_params = {
                                        k: v for k, v in params.items() if k != "clf"
                                    }
                                    mlflow.log_params(clean_params)

                
                
                best_model = random_search.best_estimator_

                metrics = self.evaluate_model(best_model,X_test,y_test)
                self.save_model(best_model)
                
                
                mlflow.set_tag(
                                "best_model_type",
                                best_model.named_steps["clf"].__class__.__name__
                            )

                
                logger.info("Logging the model to MLFLOW")
                mlflow.log_artifact(self.model_output_path)
                
                logger.info("Logging Params and metrics to mlflow")
                mlflow.log_params(best_model.get_params())
                mlflow.log_metrics(metrics)
                
                
                mlflow.sklearn.log_model(
                                best_model,
                                artifact_path="best_model"
                            )

                
                logger.info("Model training successfully completed")
            
        except Exception as e:
            logger.error (f"Error in Model training Pipeline {e}")
            raise CustomException ("Failed during Model training Pipeline", e)
        
if __name__ =="__main__":
    trainer = ModelTraining(TRAIN_FILE_PATH,TEST_FILE_PATH,MODEL_OUTPUT_PATH)
    trainer.run()