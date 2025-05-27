
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dvclive import Live
import yaml
import mlflow.sklearn
import dagshub
import mlflow
from mlflow.models import infer_signature
import seaborn as sns 
import matplotlib.pyplot as plt


# dagshub.init(repo_owner='sumitrwk90', repo_name='water-test', mlflow=True)
# # Set the experiment name in MLflow
# mlflow.set_experiment("DVC PIPELINE ")
# # Set the tracking URI for MLflow to log the experiment in DagsHub
# mlflow.set_tracking_uri("https://dagshub.com/sumitrwk90/water-test.mlflow") 


# Key based authentication
import os
# Load dagshub token
dagshub_token = os.getenv("DAGSHUB_TOKENS")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"]=dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token

# Dagshub repository details
dagshub_url = "https://dagshub.com"
repo_owner = "sumitrwk90"
repo_name = "water-test"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Final_model")

# # .env based
# from dotenv import load_dotenv
# import os

# load_dotenv()
# token = os.getenv("DAGSHUB_TOKENS")

# if token is None:
#     raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")



# Load Data
def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")
# test_data = pd.read_csv("./data/processed/test_processed.csv")

def prepare_data(data : pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")
# x_test = test_data.iloc[:,0:-1].values
# y_test = test_data.iloc[:,-1].values

def load_model(filepath : str) -> None:
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath} : {e}")
# model = pickle.load(open("model.pkl", "rb"))

def evaluation_model(model, X_test : pd.DataFrame, y_test : pd.Series, model_name: str) -> dict:
    try:

        # dvc tracking config
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]



        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)


        # dvc exp tracking config
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", pre)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators)


            # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        
        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)


        metrics_dict = {

                "accuracy":acc,
                "precision":pre,
                "recall":recall,
                "f1_score":f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")
    
def save_metrics(metrics_dict : dict, filepath : str) -> None:
    try:
        with open('reports/metrics.json', 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath} : {e}")
    
# with open('metrics.json', 'w') as file:
#     json.dump(metrics_dict, file, indent=4)

def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"

        # Define the path to the metrics file
        metrics_path = "reports/metrics.json"
        model_name = "Best Model"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        # Start MLflow run
        with mlflow.start_run() as run:
            metrics = evaluation_model(model, X_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            
            # Log the source code file
            mlflow.log_artifact(__file__)

            signature = infer_signature(X_test,model.predict(X_test))

            mlflow.sklearn.log_model(model,"Best Model",signature=signature)

            #Save run ID and model info to JSON File
            run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)

    except Exception as e:
        raise Exception(f"An Error occurred: {e}")

if __name__ == "__main__":
    main()