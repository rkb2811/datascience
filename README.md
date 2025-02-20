# EXP 1:
Loading and Merging Datasets in Python
<br>
What is Dataset Loading?
Loading a dataset means importing data from an external source into a Python environment for analysis.
In data analysis and machine learning, we often work with multiple datasets containing related information. To get a complete picture, we need to load and merge these datasets efficiently.

For example, in a business setting:

One dataset may contain customer details (ID, Name, Age).
<br>
Another dataset may contain purchase history (ID, Items Bought, Amount Spent).
<br>
By merging these datasets, we can analyze customer behavior.


import pandas as pd
df1= pd.read_csv('/content/drive/MyDrive/winequality-red.csv', delimiter=';')
df2= pd.read_csv('/content/drive/MyDrive/winequality-white.csv',delimiter=';')
print("red wine data: ")
df1['type']='red'
print(df1.head())
print("white wine data: ")
df2['type']='white'
print(df2.head())
merged= pd.concat([df1,df2],ignore_index= True)


print("combined: ")
print(merged.head())

