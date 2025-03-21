# Feature Selection Using Correlation on Wine Quality Dataset

## **Overview**
This repository demonstrates **filter-based feature selection** using the **UCI Wine Quality Dataset**.  
Filter methods use **statistical techniques** to select relevant features before training a machine learning model.

## **Dataset**
The dataset used is the **Wine Quality Dataset** from the **UCI ML Repository**.  
It contains **chemical properties** of wine and their corresponding **quality ratings**.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Features**:
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target Variable**:  
  - `quality` (Wine quality score between **0 and 10**)

## **Techniques Used**
1. **Correlation-Based Feature Selection**  
   - Computes a **correlation matrix** between all features.
   - Selects features that have a **correlation above 0.3** with the target variable (`quality`).
   - Removes features that have a low impact on wine quality.

## code
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Define column names
columns = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]

# Read the dataset
df = pd.read_csv(url, names=columns)

# Drop the ID column (not useful for feature selection)
df.drop(columns=["ID"], inplace=True)

# Encode the target variable (M = 1, B = 0)
le = LabelEncoder()
df["Diagnosis"] = le.fit_transform(df["Diagnosis"])

# Normalize features for chi-square test
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop(columns=["Diagnosis"]))
y = df["Diagnosis"]

# 1. **Feature Selection Using Correlation**
plt.figure(figsize=(12, 6))
corr_matrix = pd.DataFrame(X, columns=df.columns[1:]).corr()
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 2. **Feature Selection Using Chi-Square Test**
chi2_selector = SelectKBest(score_func=chi2, k=10)
chi2_selector.fit(X, y)
chi2_scores = pd.Series(chi2_selector.scores_, index=df.columns[1:])
chi2_scores.nlargest(10).plot(kind='barh', color='blue')
plt.title("Top Features Using Chi-Square Test")
plt.xlabel("Chi-Square Score")
plt.show()

# 3. **Feature Selection Using Mutual Information**
mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
mi_selector.fit(X, y)
mi_scores = pd.Series(mi_selector.scores_, index=df.columns[1:])
mi_scores.nlargest(10).plot(kind='barh', color='red')
plt.title("Top Features Using Mutual Information")
plt.xlabel("Mutual Information Score")
plt.show()
```
### Output:
![download (7)](https://github.com/user-attachments/assets/9a417771-8abc-4f02-a086-4ceb2627d1d1)
![download (8)](https://github.com/user-attachments/assets/d0eb71e2-83db-45a8-8a80-9e1dde9d0501)
![download (9)](https://github.com/user-attachments/assets/3caa8eda-cd17-4250-8f57-5ba04fac38b9)
