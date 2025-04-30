# Exp 11: Embedded Feature Selection

This project demonstrates how to apply **three embedded feature selection techniques** using Python and `scikit-learn` on the classic **Wine dataset**. Embedded methods perform feature selection **during the model training process**, making them efficient and model-aware.

We apply the following embedded methods:

1. **Lasso Regression (L1 Regularization)** – Shrinks less important feature coefficients to zero.
2. **Random Forest Feature Importance** – Measures how much each feature decreases impurity in decision trees.
3. **XGBoost Feature Importance** – Uses gradient boosting to evaluate how useful each feature is in making splits.

The goal is to identify and compare the most relevant features selected by each model for wine classification.


## Dataset

- **Wine Dataset** from `sklearn.datasets.load_wine()`
- 13 chemical attributes of wine
- Target: Wine cultivar class

## Requirements
Python 3.7+

scikit-learn

numpy

pandas
## Code

```
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load Wine dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Standardize features for Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1️⃣ Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_coeffs = pd.Series(lasso.coef_, index=data.feature_names)
lasso_selected = lasso_coeffs[lasso_coeffs != 0].index.tolist()

# 2️⃣ Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_importances = pd.Series(rf.feature_importances_, index=data.feature_names)
rf_selected = rf_importances[rf_importances > 0.05].index.tolist()  # Threshold can be adjusted

# 3️⃣ XGBoost Feature Importance
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_clf.fit(X, y)
xgb_importances = pd.Series(xgb_clf.feature_importances_, index=data.feature_names)
xgb_selected = xgb_importances[xgb_importances > 0.05].index.tolist()  # Threshold can be adjusted

# Display Results
print("Selected Features by Lasso Regression:")
print(lasso_selected)

print("\n Selected Features by Random Forest:")
print(rf_selected)

print("\n Selected Features by XGBoost:")
print(xgb_selected)
```
## Output
Selected Features by Lasso Regression:
['alcalinity_of_ash', 'flavanoids', 'hue', 'od280/od315_of_diluted_wines', 'proline']

 Selected Features by Random Forest:
['alcohol', 'flavanoids', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

 Selected Features by XGBoost:
['malic_acid', 'flavanoids', 'color_intensity', 'od280/od315_of_diluted_wines', 'proline']
