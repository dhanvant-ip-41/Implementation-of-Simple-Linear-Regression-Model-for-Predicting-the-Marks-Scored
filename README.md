# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries and read the csv file. Set values for variables x and y.
2. Then get the test and train data using the train_test_split() function. Now predict the y values using Linear Regression.
3. Present the graph with plots and regression line for both train and test data.
4. Calculate the MSE, MAE, RMSE for test y and predicted y.

## Program:

#### Program to implement the simple linear regression model for predicting the marks scored.
* Developed by : Dhanvant Kumar V
* RegisterNumber :  212224040070

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

```
```python
df=pd.read_csv("student_scores.csv")
print(df)
```
```python
print(df.head())
```
```python
print(df.tail())
```
```python
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
```
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)
print(y_test)
```
```python
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lr.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```python
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,lr.predict(x_test),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```python
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error = ",mse)
print("Mean Absolute Error = ",mae)
print("R0ot Mean Square Error = ",rmse)
```

## Output:
### Data Frame 
![alt text](image.png)
### Head Values
![alt text](image-2.png)
### Tail Values
![alt text](image-1.png)
### X and Y values
![alt text](image-8.png)
### Predicted Y and Test Y values
![alt text](image-4.png)
### Training Set Graph
![alt text](image-5.png)
### Test Set Graph
![alt text](image-6.png)
### MSE, MAE AND RMSE
![alt text](image-9.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
