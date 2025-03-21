# Wrapper Feature Selection (Part-2)

## Overview
This project applies **Wrapper Feature Selection (Part-2)** using **Recursive Feature Elimination with Cross-Validation (RFECV)** to identify the most important features for predicting wine quality. The dataset is obtained from the **UCI Machine Learning Repository**.

## Dataset
The dataset used is **Wine Quality Dataset** from UCI ML Repository, containing various physicochemical properties of wine samples and their quality ratings.

## Installation
To run this project, ensure you have Python installed along with the required dependencies.

## Code
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Define column names
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", 
           "slope", "ca", "thal", "target"]

# Read dataset
df = pd.read_csv(url, names=columns, na_values="?")

# Drop rows with missing values
df.dropna(inplace=True)

# Convert target to binary classification (0: No heart disease, 1: Heart disease)
df["target"] = (df["target"] > 0).astype(int)

# Define features (X) and target (y)
X = df.drop(columns=["target"])
y = df["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. **Recursive Feature Elimination with Cross-Validation (RFECV)**
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(estimator, step=1, cv=5, scoring="accuracy")
rfecv.fit(X_train, y_train)

# Get selected features
selected_features_rfecv = X.columns[rfecv.support_]
print("\nSelected Features (RFECV):", list(selected_features_rfecv))

# Evaluate RFECV model performance
y_pred_rfecv = rfecv.estimator_.predict(X_test)
print("RFECV Model Accuracy:", accuracy_score(y_test, y_pred_rfecv))

# 2. **Exhaustive Feature Selection (EFS)**
efs = EFS(estimator, min_features=3, max_features=5, scoring="accuracy", cv=3)
efs.fit(X_train, y_train)

# Get selected features
selected_features_efs = list(efs.best_feature_names_)
print("\nSelected Features (EFS):", selected_features_efs)

# Evaluate EFS model performance
y_pred_efs = estimator.fit(X_train[:, efs.best_idx_], y_train).predict(X_test[:, efs.best_idx_])
print("EFS Model Accuracy:", accuracy_score(y_test, y_pred_efs))

# Plot RFECV feature ranking
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=rfecv.ranking_, color="blue")
plt.title("Feature Ranking (RFECV)")
plt.xlabel("Features")
plt.ylabel("Ranking (Lower is Better)")
plt.xticks(rotation=45)
plt.show()
```
