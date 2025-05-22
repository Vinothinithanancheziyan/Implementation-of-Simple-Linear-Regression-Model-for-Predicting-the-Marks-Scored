# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: VINOTHINI T

RegisterNumber: 212223040245
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
### Head Values
![Image](https://github.com/user-attachments/assets/9e287af0-ff4a-4350-afff-12a9164ad37a)

### Tail Values
![Image](https://github.com/user-attachments/assets/27ea04aa-99f5-4f3c-99db-90c37f332f54)

### X Values
![Image](https://github.com/user-attachments/assets/52faf9f8-f435-4a58-935d-e4d89fc95e3c)


### y Values
![Image](https://github.com/user-attachments/assets/54e0a398-ffb0-40da-89c7-d5a78394473f)

### Predicted Values
![Image](https://github.com/user-attachments/assets/9c2a67f6-156e-4bd9-acfa-589e386f4b67)


### Actual Values
![Image](https://github.com/user-attachments/assets/1334df3c-b63e-42ad-a8cf-6c15b292a7bd)

### Training Set
![Image](https://github.com/user-attachments/assets/53590d4e-3a9c-44cd-b44d-448f7308a382)

### Testing Set
![Image](https://github.com/user-attachments/assets/885c9d55-6bf8-41a9-bcec-5ead0f3153af)

### MSE, MAE and RMSE
![Image](https://github.com/user-attachments/assets/c6706544-e259-47df-88c7-e0e4677a2196)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
