# Wrapper Feature Selection (Part-2)

## Overview
This project applies **Wrapper Feature Selection (Part-2)** using **Recursive Feature Elimination with Cross-Validation (RFECV)** to identify the most important features for predicting wine quality. The dataset is obtained from the **UCI Machine Learning Repository**.

## Dataset
The dataset used is **Wine Quality Dataset** from UCI ML Repository, containing various physicochemical properties of wine samples and their quality ratings.

## Installation
To run this project, ensure you have Python installed along with the required dependencies.

## code
```
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/Datasets/winequality-red.csv', delimiter=';')

# Prepare data
X = df.drop(columns=['quality'])
y = df['quality']

# Scale data
X = StandardScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply RFECV
model = LogisticRegression(max_iter=2000)
rfecv = RFECV(estimator=model, cv=3, scoring='accuracy')
rfecv.fit(X_train, y_train)

# Display results
selected_features = df.drop(columns=['quality']).columns[rfecv.support_]
print("Optimal number of features:", rfecv.n_features_)
print("Selected Features:", selected_features.tolist())
