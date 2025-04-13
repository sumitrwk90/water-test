import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dagshub
from sklearn.model_selection import train_test_split

import dagshub
dagshub.init(repo_owner='sumitrwk90', repo_name='water-test', mlflow=True)
mlflow.set_experiment("Experiment 3.0")
mlflow.set_tracking_uri("https://dagshub.com/sumitrwk90/water-test.mlflow") 

# Load and preprocess data
data = pd.read_csv(r"C:\Users\Lenovo\water-potability-prediction\data_storage\water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
    return df

# Fill missing values with median
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_test = test_processed_data["Potability"]

# Define the baseline models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
     "XG Boost" : XGBClassifier()
 }

# Start a single MLflow run for the entire experiment
with mlflow.start_run(run_name="Water Potability Models Experiment"):
    # Loop through each baseline model
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):  # Create a child run for each model
            model.fit(X_train, y_train)
            
            # Save model
            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_filename, "wb"))
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
            
            # Log artifacts (confusion matrix)
            mlflow.log_artifact(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
            
            # Log the model
            mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))
    
    # Log source code
    mlflow.log_artifact(__file__)
    
    # Add tags
    mlflow.set_tag("author", "sumit kumar")
    # mlflow.set_tag("model", "model_name")
    
    print("All models have been trained and logged as child runs successfully.")