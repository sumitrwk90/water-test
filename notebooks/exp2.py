# Import necessary libraries for data manipulation, ML models, metrics, visualization, logging, and tracking
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

# Initialize DagsHub integration and set the experiment for MLflow tracking
dagshub.init(repo_owner='sumitrwk90', repo_name='water-test', mlflow=True)
mlflow.set_experiment("Experiment 2")  # Set the experiment name
mlflow.set_tracking_uri("https://dagshub.com/sumitrwk90/water-test.mlflow")   # Registry URI for model management

# Load the dataset from CSV file and split into training and testing sets
data = pd.read_csv(r"C:\Users\Lenovo\water-potability-prediction\data_storage\water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Define a function to fill missing values with the median value for each column
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():  # Check if there are missing values
            median_value = df[column].median()  # Calculate the median for the column
            df[column].fillna(median_value, inplace=True)  # Fill missing values with the median
    return df

# Preprocess training and testing data to fill missing values
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Split the data into features (X) and target (y) for training and testing
X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_test = test_processed_data["Potability"]

# Define multiple baseline models to compare performance
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XG Boost": XGBClassifier()
}

# Start a parent MLflow run to track the overall experiment
with mlflow.start_run(run_name="Water Potability Models Experiment"):
    # Iterate over each model in the dictionary
    for model_name, model in models.items():
        # Start a child run within the parent run for each individual model
        with mlflow.start_run(run_name=model_name, nested=True):
            # Train the model on the training data
            model.fit(X_train, y_train)
            
            # Save the trained model using pickle
            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_filename, "wb"))
            
            # Make predictions on the test data
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log the calculated metrics to MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Generate and visualize the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Create a heatmap of the confusion matrix
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png")  # Save the plot
            
            # Log the confusion matrix plot as an artifact in MLflow
            mlflow.log_artifact(f"confusion_matrix_{model_name.replace(' ', '_')}.png")
            
            # Log the model to MLflow
            mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))
    
            # Log the source code file for reproducibility (the current script)
            mlflow.log_artifact(__file__)
    
            # Set tags for the run to provide additional metadata
            mlflow.set_tag("author", "sumit kumar")
    
    print("All models have been trained and logged as child runs successfully.")