# Wrapper Feature Selection (RFE)

## Overview  
Feature selection is a crucial step in machine learning that helps improve model performance by eliminating irrelevant or redundant features.  
This project demonstrates the **Wrapper Feature Selection** technique using **Recursive Feature Elimination (RFE)** with **Logistic Regression**.  
The goal is to select the top **5 most significant features** from the **Wine Quality Dataset** obtained from the **UCI ML Repository**.  

## Description  
Recursive Feature Elimination (RFE) is a wrapper method that selects the most relevant features by recursively training a model and eliminating the least important features.  
In this project, **Logistic Regression** is used as the base model for RFE to determine the most influential features in predicting wine quality.  

## How to Use  
1. Download the dataset and save it in the correct location.  
2. Copy and run the Python script in your local environment or a Jupyter Notebook.  
3. Ensure you have installed the required libraries (`pandas`, `sklearn`).  
4. The script will output the **top 5 selected features**.  

# Code  
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset from UCI repository
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Define column names
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Read the dataset
df = pd.read_csv(url, names=columns)

# Define features (X) and target (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split data into training and testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1️⃣ **Recursive Feature Elimination (RFE)**
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_train, y_train)

# Get selected features
selected_features_rfe = X.columns[rfe.support_]
print("\nSelected Features (RFE):", list(selected_features_rfe))

# Train & evaluate model with selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)
model.fit(X_train_rfe, y_train)
y_pred_rfe = model.predict(X_test_rfe)
print("RFE Accuracy:", accuracy_score(y_test, y_pred_rfe))

# 2️⃣ **Sequential Feature Selection (Forward)**
sfs_forward = SequentialFeatureSelector(model, n_features_to_select=5, direction="forward")
sfs_forward.fit(X_train, y_train)
selected_features_sfs_fwd = X.columns[sfs_forward.get_support()]
print("\nSelected Features (Forward SFS):", list(selected_features_sfs_fwd))

# 3️⃣ **Sequential Feature Selection (Backward)**
sfs_backward = SequentialFeatureSelector(model, n_features_to_select=5, direction="backward")
sfs_backward.fit(X_train, y_train)
selected_features_sfs_bwd = X.columns[sfs_backward.get_support()]
print("\nSelected Features (Backward SFS):", list(selected_features_sfs_bwd))
```
### Output:
Selected Features (RFE): ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
RFE Accuracy: 0.7727272727272727
