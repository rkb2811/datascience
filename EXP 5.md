# EXPERIMENT 5 
## Introduction 
This repository contains the code and results for the foyrth experiment in the Data Science Fundamentals with Python course. The objective of this experiment is to data visualization using matplot library part-1 using data set obtained from UCIML repository 
## Steps-

 1.Import Libraries:

Import necessary libraries: pandas for data handling, matplotlib for plotting, and seaborn for advanced visualizations.

 2.Load Dataset:
Load the dataset using pd.read_csv(), specifying the correct delimiter (semicolon ; in this case).

 3.Clean Column Names:
Strip any leading or trailing spaces from the column names to avoid reference issues when accessing columns.

4.Check Data:

Inspect the first few rows (head()) and column names (columns) to understand the dataset structure.

 5.Handle Missing Values:

Fill any missing values in numeric columns with the mean of each respective column.
### 6.Plotting 
 a) Histogram:

Plot a histogram of the alcohol column to visualize its distribution.
<br>
Code:
```
print("HISTOGRAM")
plt.figure(figsize=(8, 5))
plt.hist(df['sepal_length'], bins=15, color='blue',  alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Length')
```

 b) Box Plot:

Create a box plot for the fixed acidity column to identify its distribution and potential outliers.
<br>
Code:
```
print("BOX PLOT")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, palette="Set2")
plt.title("Box Plot of Iris Dataset Features", fontsize=14)
plt.show()
```

 c) Scatter Plot:
Plot a scatter plot between fixed acidity and citric acid to see their relationship.
<br>
Code:
```
print("SCATTER PLOT")
plt.figure(figsize=(8, 5))
plt.scatter(df['sepal_length'], df['sepal_width'], c='red', alpha=0.6)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.grid(True)
plt.show()
```

 d) Correlation Heatmap:

Generate a heatmap to show the correlation between different numeric features in the dataset.
<br>
Code:
```
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


```
### 7.Data Insights:

Check for missing values in the dataset and print the summary statistics for the numeric columns.


## Concepts used :

-Data Loading: Load the dataset from a CSV file.

-Data Cleaning: Clean column names and fill missing values with the mean.

-Data Visualization:
-Histograms: Show distribution of a feature.
-Box Plots: Show spread and outliers.
-Scatter Plots: Show relationships between two features.
-Correlation Heatmap: Show correlations between features.

## Example Code:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from UCI ML repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)

# Display first 5 rows
print(df.head())

# Histogram
print("HISTOGRAM")
plt.figure(figsize=(8, 5))
plt.hist(df['sepal_length'], bins=15, color='blue',  alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.title('Histogram of Sepal Length')

plt.show()


# Scatter Plot
print("SCATTER PLOT")
plt.figure(figsize=(8, 5))
plt.scatter(df['sepal_length'], df['sepal_width'], c='red', alpha=0.6)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.grid(True)
plt.show()
# correlation heatmap
corr_matrix = df.corr(numeric_only=True)
# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# Box Plot to visualize outliers
print("BOX PLOT")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, palette="Set2")
plt.title("Box Plot of Iris Dataset Features", fontsize=14)
plt.show()



```
![download (3)](https://github.com/user-attachments/assets/ef627e22-4232-42ef-88bf-3a639a4c9f48)
![download (2)](https://github.com/user-attachments/assets/66d2778a-74c4-440d-b85c-f2d9f126d432)
![download (4)](https://github.com/user-attachments/assets/40991187-dbfa-4870-a109-aa3febd98746)
![download (1)](https://github.com/user-attachments/assets/9737d699-6262-457c-a51a-256b4e8b1f06)
