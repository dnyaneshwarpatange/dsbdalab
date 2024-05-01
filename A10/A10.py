# Assignment number 10:

# Load the Iris dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Display and iterpret Boxplot & Histogram
# Identify outliers

#---------------------------------------------------------------------------------------
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)
df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')
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
# Display and fill the Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
fig, axis = plt.subplots(2,2)
sns.boxplot(ax = axis[0,0], data = df, y='SL')
sns.boxplot(ax = axis[0,1], data = df, y='SW')
sns.boxplot(ax = axis[1,0], data = df, y='PL')
sns.boxplot(ax = axis[1,1], data = df, y='PW')
plt.show()
fig, axis = plt.subplots(2,2)
sns.boxplot(ax = axis[0,0], data = df, y='SL', hue='Species')
sns.boxplot(ax = axis[0,1], data = df, y='SW', hue='Species')
sns.boxplot(ax = axis[1,0], data = df, y='PL', hue='Species')
sns.boxplot(ax = axis[1,1], data = df, y='PW', hue='Species')
plt.show()
fig, axis = plt.subplots(2,2)

sns.histplot(ax = axis[0,0], data = df, x='SL', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[0,1], data = df, x='SW', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1,0], data = df, x='PL', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1,1], data = df, x='PW', multiple = 'dodge', shrink = 0.8, kde = True)
plt.show()
fig, axis = plt.subplots(2,2)
sns.histplot(ax=axis[0,0], data=df, x='SL', hue='Species', element='poly', shrink=0.8, kde= True)
sns.histplot(ax=axis[0,1], data = df, x='SW', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,0], data = df, x='PL', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,1], data = df, x='PW', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
plt.show()