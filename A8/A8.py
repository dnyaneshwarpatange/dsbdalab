import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#print(sns.get_dataset_names())
df=sns.load_dataset('titanic')
#print(df)

# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)

df.head()
sns.set_style('whitegrid')
sns.displot(df['fare'] ,kde=True)
plt.show()
sns.displot(df['fare'], kde = False)
plt.show()
sns.displot(df['fare'], kde = False, bins = 10)
plt.show()
