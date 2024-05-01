import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,confusion_matrix,precision_recall_fscore_support
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df= pd.read_csv('Social_Network_Ads.csv')
print(df)
#select features and target variables

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

print(X)
print(y)
# fit scaler on training data
norm = MinMaxScaler().fit(X)
# transform training data
X = norm.transform(X)

#splitting data into training and testing set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(y_train.shape)


#create a linear regression model

model=LogisticRegression()

#fit the model on training data
model.fit(X_train,y_train)

#make prediction on the test data

y_pred = model.predict(X_test)
print(y_pred)

# evaluate the model

mse = mean_squared_error(y_test,y_pred)

print(mse)

r_mse = np.sqrt(mse)

print(r_mse)

plt.scatter(y_test,y_pred,color='k')
plt.title('Logistic Regression Visualization')
plt.xlabel('actual values')
plt.ylabel('predicted values')

cf = confusion_matrix(y_test, y_pred)
print(cf)

score = precision_recall_fscore_support(y_test, y_pred, average='micro')
print('Precision of Model: ', score[0])
print('Recall of Model: ', score[1])
print('F-Score of Model: ', score[2])

sns.heatmap(cf,annot=True,fmt = 'd',cmap='Blues')
plt.xlabel("Acutual value")
plt.ylabel("predicted values")
plt.title("Confusion Matrics")
plt.show();

#
