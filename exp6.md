# Iris Dataset Visualization using Matplotlib and Seaborn
## Project Overview
This project demonstrates various data visualization techniques using the Iris dataset from the UCI Machine Learning Repository. The dataset consists of three species of Iris flowers (Setosa, Versicolor, Virginica) with four numerical features (sepal length, sepal width, petal length, petal width).

## Dataset Information
Dataset Name: iris.data
</br>
Source: UCI Machine Learning Repository
</br>
Format: CSV
</br>
Number of Instances: 150
</br>
Number of Attributes: 4 numerical + 1 categorical (species)
## Column Names
### Feature	Description
sepal_length = Sepal length in cm
</br>
sepal_width	= Sepal width in cm
</br>
petal_length	= Petal length in cm
</br>
petal_width =	Petal width in cm
## code for Advanced Visualizations

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Define column names
columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"]

# Read the dataset
df = pd.read_csv(url, names=columns)

# Display first few rows
print(df.head())

# Create a scatter plot for Sepal Length vs Sepal Width
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Sepal Length"], y=df["Sepal Width"], hue=df["Species"], palette="coolwarm")
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Species")
plt.show()

# Create a histogram for Petal Length
plt.figure(figsize=(8, 5))
sns.histplot(df["Petal Length"], bins=20, kde=True, color="purple")
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Count")
plt.show()

# Create a boxplot for Sepal Width
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Species"], y=df["Sepal Width"], palette="Set2")
plt.title("Boxplot of Sepal Width by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.show()
```
### Output:
```
Sepal Length  Sepal Width  Petal Length  Petal Width      Species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
```
![download (5)](https://github.com/user-attachments/assets/26204a3c-a667-4150-9a47-486dfe019b45)
![download (6)](https://github.com/user-attachments/assets/acbc78a4-933b-4ae0-a422-fd2257ff68c8)
