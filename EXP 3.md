# EXP 3: Data preprocessing- Applying encoding techniques. 
<br>
Encoding techniques are essential for converting categorical variables into numerical format for use in machine learning algorithms.
<br>
Some common encoding techniques used in this experiment are:
<br>
Label Encoding – Each category is given a unique number (e.g., Apple = 0, Banana = 1, Orange = 2).
<br>
One-hot Encoding – Creates separate columns for each category and marks them as 1 (present) or 0 (not present).
<br>
Ordinal Encoding – Assigns numbers to categories based on their order (e.g., Small = 1, Medium = 2, Large = 3).
<br>
Target Encoding – Replaces categories with the average value of the target variable for that category.
<br>
Binary Encoding – Converts category values into binary (0s and 1s) and stores them in multiple columns.
<br>
Frequency Encoding – Replaces categories with how often they appear in the dataset.
<br>
Below is my code with corresponding comments for you to understand 
<br>

```
 #EXP 3: ENCODING
#1. label encoding
from sklearn.preprocessing import LabelEncoder
data=['Low','High','Medium','High','Medium']
encoder= LabelEncoder()
encoded_data= encoder.fit_transform(data)
print(f"Label encoded data: {encoded_data}")
#2. one hot encoding
import pandas as pd
data=['Red','Blue','Green','Blue','Red']
df= pd.DataFrame(data,columns=['Color'])
one_hot_encoded=pd.get_dummies(df['Color'])
print("one hot encoded: \n")
print(one_hot_encoded)
#3. ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
data=[['Low'],['High'],['Medium'],['High'],['Medium']]
encoder= OrdinalEncoder(categories=[['Low','Medium','High']])
encoded_data=encoder.fit_transform(data)
print(f"Ordinal Encoded Data: {encoded_data}")
#4. Target encoding
!pip install category_encoders
import pandas as pd
import category_encoders as ce
data= {'Color':['Red','Blue','Green','Blue','Red','Blue','Green','Green','Green','Blue'],'Target':['1','0','0','1','1','1','0','1','0','1']}
df=pd.DataFrame(data)
df['Target'] = df['Target'].astype(int)
encoder= ce.TargetEncoder(cols=['Color'])
encoded_data= encoder.fit_transform(df['Color'],df['Target'])
print(f"Target encoded: {encoded_data}")
#5. binary encoding
import category_encoders as ce
data=['Red','Green','Blue','Red','Grey']
encoder = ce.BinaryEncoder(cols=['Color'])
encoded_data= encoder.fit_transform(pd.DataFrame(data,columns=['Color']))
print("binary encoded: \n")
print(encoded_data)
#6. frequency encoding
import pandas as pd
data=['Red','Green','Blue','Red','Red']
series_data= pd.Series(data)
frequency_encoding= series_data.value_counts()
encoded_data= [frequency_encoding[x] for x in data]
print("frequency encoded: \n")
print("encoded data: ",encoded_data)
