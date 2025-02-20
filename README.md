# EXP 2: Data Preprocessing - Cleaning dataset
<br>
Steps-
<br>
1.Loading datasets: The datasets are loaded using the Pandas library from the UCI ML Repository.
<br>
2.Handling Missing Values:Handle missing values by removing them or replacing Null/NaN with defaults, statistics, or predictions.
<br>
3.Removing Duplicates: Remove duplicates by dropping them or keeping the first/last occurrence based on data needs.
<br>
4.Handling Outliers:Handle outliers by removing or capping values beyond the IQR range (1.5×IQR rule).
<br>
<br>
Concepts Used-
<br>
Pandas Library: Used for data manipulation and analysis.
<br>
DataFrames: Data structures provided by Pandas to store and manipulate tabular data.
<br>
Data Cleaning: Techniques such as removing duplicates, handling missing values, and standardizing data.
<br>
<br>
Step 1:  Import necessary library
<br>
Pandas is a powerful Python library for data manipulation, analysis, and processing using DataFrames and Series.
```python 
import pandas as pd
```
