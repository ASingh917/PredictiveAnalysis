# Credit Card Fraud Detection
This project analyzes a credit card fraud dataset using Python

# Steps

# Step 1: Import Libraries

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
# Step 2: Load the Dataset
```
data = pd.read_csv('Creditcard_data.csv')
```
# Step 3: Initial Exploration of the Dataset
```
data.head()
data.info()
data.describe()
```
data.head(): To display the first 5 rows of the dataset.

data.info(): Provides metadata like column names, data types, and non-null counts.

data.describe(): Provides descriptive analysis of the columns like mean,median etc .

# Step 4: Class Distribution Analysis
```
data["Class"].value_counts()
```
Count the number of instances having target variable (class) as class == 1 or class == 0.

# Step 5: Missing Values Check
```
missing_values = data.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)
```
To check whether there is any missing value or not in any of the columns.

# Step 6: Separate Classes
```
data_0 = data[data['Class'] == 0]
data_1 = data[data['Class'] == 1]
print('class 0:', data_0.shape)
print('class 1:', data_1.shape)
```
Divides the dataset into two subsets:

data_0: Transactions labeled as class 0 (non-fraud).

data_1: Transactions labeled as class 1 (fraud).

# Step 7: Visualize Target Variable
```
data['Class'].value_counts().plot(kind='bar', color='skyblue', title="Target Variable Distribution")
```
Creates a bar graph to visualize the imbalance between class 0 and class 1.

# Step 8: Handle Imbalanced Dataset with SMOTE
```
from imblearn.over_sampling import SMOTE
from collections import Counter

y = data['Class']
x = data.drop('Class', axis=1)

smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x, y)
```
SMOTE (Synthetic Minority Oversampling Technique): Balances the dataset by creating synthetic samples for the minority class.

x: Independent features.

y: Target variable.

fit_resample: Applies SMOTE to create x_smote and y_smote, balanced datasets.

