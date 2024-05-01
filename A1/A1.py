# Assignment number 1:
# Load the dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Change datatype of variable
# Quantization (Encoding): Convert categorical to numerical variable
# Normalization

#---------------------------------------------------------------------------------------
# Importing libraries
import pandas as pd
import numpy as np 
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('Placement.csv')
#---------------------------------------------------------------------------------------
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Fill the missing values
df['gender'].fillna(df['gender'].mode()[0])
df['ssc_p'].fillna(df['ssc_p'].mean())
print('Mode of ssc_b: ', df['ssc_b'].mode())
df['ssc_b'].fillna(df['ssc_b'].mode()[0])
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# changing data type of columns
# see the datatype using df.dtypes
# change the datatype using astype
df['sl_no']=df['sl_no'].astype('int8')
print('Change in datatype: ', df['sl_no'].dtypes)

#---------------------------------------------------------------------------------------
# Converting categorical (qualitative) variable to numeric (quantitative) variable

# 1. Find and replace method
# 2. Label encoding method
# 3. OrdinalEncoder using scikit-learn

# Find and replace method
df['gender'].replace(['M','F'],[0,1])
# Label encoding method
df['ssc_b']=df['ssc_b'].astype('category') #change data type to category
df['ssc_b']=df['ssc_b'].cat.codes
# Ordinal encoder using Scikit-learn
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df[['hsc_b']]=enc.fit_transform(df[['hsc_b']])
print('After converting categorical variable to numeric variable: ')
print(df.head().T)
#---------------------------------------------------------------------------------------
# Normalization of data
# converting the range of data into uniform range
# marks [0-100] [0-1]
# salary [200000 - 200000 per month] [0-1]
# Min-max feature scaling
# minimum value = 0
# maximum value = 1
# when we design model the higher value over powers in the model
df['salary']=(df['salary']-df['salary'].min())/(df['salary'].max()-df['salary'].min())
# (x - min value into that column)/(max value - min value)
# Maximum absolute scaler using scikit-learn
from sklearn.preprocessing import MaxAbsScaler
abs_scaler=MaxAbsScaler()
df[['mba_p']]=abs_scaler.fit_transform(df[['mba_p']])
#---------------------------------------------------------------------------------------
print(df.head().T)