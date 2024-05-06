# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```py
# Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
# Developed by: Sanjay Ragavendar M K
# RegisterNumber:  212222100045
```
```py
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
```
```py
data.info()
```
```py
data.isnull().sum()
```
```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
```
```py
x=data[['Position','Level']]
x
```
```py
y=data['Salary']
y
```
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```
```py
from sklearn.tree import DecisionTreeClassifier,plot_tree
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```py
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
```
```py
r2=metrics.r2_score(y_test,y_pred)
r2
```
```py
import matplotlib.pyplot as plt
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:

![Screenshot 2024-04-06 114035](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/146315115/69666c35-6fac-4e88-8d42-0119bdb0ce89)

![Screenshot 2024-04-06 114043](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/146315115/2de5b878-0b66-42c0-bac7-cb3a9b6fba0d)

![Screenshot 2024-04-06 114047](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/146315115/19e992d1-f22c-4492-a088-760d25dc607f)

![Screenshot 2024-04-06 114051](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/146315115/17415255-3f43-4983-8cd6-337e31da5153)

![Screenshot 2024-04-06 114055](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/146315115/8e9b12d9-c5a6-4608-ab3a-eea15dc71a74)

![Screenshot 2024-04-06 114058](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/146315115/798096c3-f0a1-4a08-8643-68ed09c7d2e0)

![Screenshot 2024-04-06 114103](https://github.com/rohithprem18/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/146315115/7c815154-c5c4-423a-b089-c2aed30ab908)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
