import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_params(filepath : str) -> float:

    """General Exception handelar"""

    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath} : {e}")

# # connect params.yaml file
# test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

def load_data(filepath : str) -> pd.DataFrame:
    
    """Use general exception handelar"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")

# data = pd.read_csv(r"C:\Users\Lenovo\ml_pipeline\data_storage\water_potability.csv")

def split_data(data : pd.DataFrame, test_size : float) -> tuple[pd.DataFrame, pd.DataFrame]:

    """ValueError"""
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except ValueError as e:
        raise ValueError(f"Error Splitting data : {e}")

# train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

def save_data(df : pd.DataFrame, filepath : str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} : {e}")


def main():
    data_filepath = "./data_storage/water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw")
# data_path = os.path.join("data", "raw")

    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)

        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
    # os.makedirs(data_path)

        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred : {e}")
# train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

if __name__ == "__main__":
    main()