# Assignment number 3:
# Load the dataset
# Display basic information
# Display null values
# Fill the null values
# Display overall statistical information
# Display groupwise statistical information
#---------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
df = pd.read_csv('Employee_Salary.csv')
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
# Display Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Fill the missing values
df['Gender'].fillna(df['Gender'].mode()[0])
df['Experience'].fillna(df['Experience'].mean())
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Display Overall Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# groupwise statistical information
print('Groupwise Statistical Summary....')
print('\n-------------------------- Experience -----------------------\n')
print(df['Experience'].groupby(df['Gender']).describe())
print('\n-------------------------- Age -----------------------\n')
print(df['Age'].groupby(df['Gender']).describe())
print('\n-------------------------- Salary -----------------------\n')
print(df['Salary'].groupby(df['Gender']).describe())
#---------------------------------------------------------------------------------------
df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)

df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')
print(df.head().T)
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
print('Groupwise Statistical Summary....')
print('\n-------------------------- Sepal Length -----------------------\n')
print(df['SL'].groupby(df['Species']).describe())
print('\n-------------------------- Sepal Width -----------------------\n')
print(df['SW'].groupby(df['Species']).describe())
print('\n-------------------------- Petal Length -----------------------\n')
print(df['PL'].groupby(df['Species']).describe())
print('\n-------------------------- Petal Width -----------------------\n')
print(df['SW'].groupby(df['Species']).describe())