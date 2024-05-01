import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("***** Housing dataset*****")
df=pd.read_csv('Boston.csv')
print("------Data-------")
print(df)
print("--------NUll value counts----")
print(df.isnull().sum())
df['crim'].fillna(int(df['crim'].mean()))
df['zn'].fillna(int(df['zn'].mean()))
df['indus'].fillna(int(df['indus'].mean()))
df['chas'].fillna(int(df['chas'].mean()))
df['age'].fillna(int(df['age'].mean()))
df['lstat'].fillna(int(df['lstat'].mean()))
print("--------NUll value count After filling null values----")
print(df.isnull().sum())
# feature for prediction(you can modify based on you requirements )
X= df[['rm','lstat','crim']]
Y= df['medv']
# splits data into training and testing sets

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)
# create a linear regresion model
model=LinearRegression()
#Train the model on training set
model.fit(X_train,Y_train)
#Make prediction on the test set
Y_pred=model.predict(X_test)
print(Y_pred)
#Evalute the model
mse=mean_squared_error(Y_test,Y_pred)
r2=r2_score(Y_test,Y_pred)
print(f'Mean Squared Error:{mse}')
print(f'R-squared:{r2}')
#plot the  regression line
plt.scatter(Y_test,Y_pred)
plt.plot([min(Y_test),max(Y_test)],(min(Y_test),max(Y_test)),linestyle="--",color="red",linewidth=2)
plt.title("linear regression model for home prices")
plt.xlabel("actual Prices")
plt.ylabel("predicted prices")
plt.show()
