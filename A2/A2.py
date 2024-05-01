import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Students data.csv')

print("===========IsNa=============")
print(df.isna().sum())

print("===========IsNull===========")
print(df.isnull().sum())

print("===========NotNull===========")
print(df.notnull().sum())

print("============FillNa============")
df['class'].fillna('A')
print(df)

print("===========Replace============")
df['class'].replace(['A','B'],[0,1])
print(df)

print("==========Rename==================")
df.rename(columns={'class':'Div'})
print(df)

print("===========DropNa============")
df.dropna(how='all')
print(df)


print("===========Z-score===========")
race=df['race']
#print(race)
mean=np.mean(race)
std=np.std(race)
print("Mean:",mean)
print("Standard deviation:",std)

threshold=3
outlier=[]
for i in race:
	z=(i-mean)/std
	if z>threshold:
		outlier.append(i)
print("Outlier :",outlier)

print("==========IQR==========")
#print(race)
algebra=df['Statistics']
#Nrace=sorted(race)
#print(Nrace)

q1,q3=np.percentile(algebra,[25,75])
print("Q1,Q3:",q1,q3)

iqr=q3-q1
print("IQR:",iqr)

lower_fence=q1-(1.5*iqr)
upper_fence=q3+(1.5*iqr)
print("Lower fence,upper_fence:",lower_fence,upper_fence)

outlier=[]
for x in algebra:
	if((x>upper_fence)or(x<lower_fence)):
		outlier.append(x)
print('Outliner in the dataset is',outlier)
fig=plt.figure(figsize=(10,7))
plt.boxplot(df['Statistics'])
plt.show()	

ua=np.where(df['Statistics']>=upper_fence)[0]
la=np.where(df['Statistics']<=lower_fence)[0]
df.drop(index=ua)
df.drop(index=la)
print("***********After removing outliner**********")
print(df['Statistics'])

print("**********Data Transformation**********")
df['Log_attendance']=np.log(df['Statistics'])
print('**Display dataset after data transformation**')
print(df)