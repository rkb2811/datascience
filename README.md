# EXP 1:
Loading and Merging Datasets in Python
<br>
What is Dataset Loading?
<br>
Loading a dataset means importing data from an external source into a Python environment for analysis.
In data analysis and machine learning, we often work with multiple datasets containing related information. To get a complete picture, we need to load and merge these datasets efficiently.

For example, in a business setting:

One dataset may contain customer details (ID, Name, Age).
<br>
Another dataset may contain purchase history (ID, Items Bought, Amount Spent).
<br>
By merging these datasets, we can analyze customer behavior.

Step 1: Import necessary library
<br>

Pandas is a powerful Python library for data manipulation, analysis, and processing using DataFrames and Series.
```
import pandas as pd
```
Step 2: Load the datasets 
<br>
Here, I've taken the wine dataset  
```
df1= pd.read_csv('/content/drive/MyDrive/winequality-red.csv', delimiter=';')
df2= pd.read_csv('/content/drive/MyDrive/winequality-white.csv',delimiter=';')
```
Step 3: Inspect the datasets 
<br>
Show the first few rows of each dataset to verify they loaded correctly.
```
print("red wine data: ")
df1['type']='red'
print(df1.head())
print("white wine data: ")
df2['type']='white'
print(df2.head())
```
Step 4: Merge the Datasets
Merge the datasets on a common column (e.g., 'ID').
```
merged= pd.concat([df1,df2],ignore_index= True)
```
Step 5: Inspect the merged Dataset
<br>
Display the first few rows of the merged dataset.
```
print("combined: ")
print(merged.head())
```
