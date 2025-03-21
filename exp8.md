# Feature Selection Using Chi-Square Test on Wine Quality Dataset

## **Overview**
This repository demonstrates **filter-based feature selection** using the **Chi-Square test** on the **UCI Wine Quality Dataset**.

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
1. **Chi-Square Test for Feature Selection**  
   - Measures the dependence between **features** and **target variable (`quality`)**.  
   - Selects the **top K features** with the highest dependency on the target.

# Code
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from scipy.stats import spearmanr

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Read dataset
df = pd.read_csv(url, delimiter=';')

# Display dataset information
print(df.head())

# Define features (X) and target (y)
X = df.drop(columns=["quality"])
y = df["quality"]

# 1. **Variance Threshold Method**
var_thresh = VarianceThreshold(threshold=0.01)  # Remove low variance features
X_var = var_thresh.fit_transform(X)

# Print selected features
selected_features_var = X.columns[var_thresh.get_support()]
print("\nSelected Features (Variance Threshold):", list(selected_features_var))

# 2. **ANOVA F-Test**
anova_selector = SelectKBest(score_func=f_classif, k=8)  # Select top 8 features
anova_selector.fit(X, y)
anova_scores = pd.Series(anova_selector.scores_, index=X.columns)

# Plot ANOVA Scores
plt.figure(figsize=(10, 5))
anova_scores.nlargest(8).plot(kind="barh", color="green")
plt.title("Top Features Using ANOVA F-Test")
plt.xlabel("F-Score")
plt.show()

# 3. **Spearman’s Rank Correlation**
spearman_corr = {col: spearmanr(X[col], y).correlation for col in X.columns}

# Convert to Series and plot
spearman_corr_series = pd.Series(spearman_corr).abs().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
spearman_corr_series.nlargest(8).plot(kind="barh", color="purple")
plt.title("Top Features Using Spearman Correlation")
plt.xlabel("Spearman Rank Correlation")
plt.show()
```
### Output:
```fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
0            7.4              0.70         0.00             1.9      0.076   
1            7.8              0.88         0.00             2.6      0.098   
2            7.8              0.76         0.04             2.3      0.092   
3           11.2              0.28         0.56             1.9      0.075   
4            7.4              0.70         0.00             1.9      0.076   

   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
0                 11.0                  34.0   0.9978  3.51       0.56   
1                 25.0                  67.0   0.9968  3.20       0.68   
2                 15.0                  54.0   0.9970  3.26       0.65   
3                 17.0                  60.0   0.9980  3.16       0.58   
4                 11.0                  34.0   0.9978  3.51       0.56   

   alcohol  quality  
0      9.4        5  
1      9.8        5  
2      9.8        5  
3      9.8        6  
4      9.4        5  

Selected Features (Variance Threshold): ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']
```
![download (10)](https://github.com/user-attachments/assets/0523db9b-1a79-4ed7-994e-d1ad073c6daa)
![download (11)](https://github.com/user-attachments/assets/7c44abea-6aba-4dcf-bcab-06a1f91243f9)
