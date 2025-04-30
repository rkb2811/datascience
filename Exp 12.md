# Supervised Classifiers on Wine Dataset
## Introduction
This project demonstrates how to apply **multiple supervised machine learning classifiers** on the classic **Wine dataset** from `scikit-learn`. The goal is to compare the performance of different algorithms for classifying wines into one of three categories based on chemical properties.

We apply and evaluate the following classifiers:

1. **Logistic Regression** 
2. **Support Vector Machine (SVM)** 
3. **K-Nearest Neighbors (KNN)** 
4. **Decision Tree Classifier** 
5. **Random Forest Classifier**.

## Explanation:
Logistic Regression: A linear classifier that works well for linearly separable classes.

Support Vector Machine (SVM): A powerful classifier, especially for high-dimensional data, works well for non-linear decision boundaries.

K-Nearest Neighbors (KNN): A simple algorithm that classifies based on the majority class of nearest neighbors.

Decision Tree: A tree-based classifier that splits data into subsets based on feature values.

Random Forest: An ensemble of decision trees, which is generally more robust than a single decision tree.
This project compares the performance of these classifiers on the Wine dataset to identify the most effective model for this task.

## Requirements
Python 3.7+

scikit-learn

numpy

pandas





##  Code

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Wine dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Support Vector Machine (SVM)": SVC(),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate models
results = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the results
    results[model_name] = accuracy

# Display the results
print("Classifier Performance on Wine Dataset:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

```
## Output
Classifier Performance on Wine Dataset:<br>
Logistic Regression: 0.9722<br>
 Support Vector Machine (SVM): 0.9722<br>
 K-Nearest Neighbors (KNN): 0.9444<br>
 Decision Tree: 0.9722<br>
Random Forest: 1.0000<br>
