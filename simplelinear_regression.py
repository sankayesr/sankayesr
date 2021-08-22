# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:20:55 2020

@author: MANSI
"""
# Machine Learning: 1 . Simple Linear Regression

# import libraries 

import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd

#import data set

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
 
# split the data into train and test data set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# fitting in the simple linear regression test model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

# Testing the training set

y_pred= regressor.predict(X_test)

# Visualization of the training set result 

mpl.scatter(X_train,y_train,color='red')

mpl.plot(X_train,regressor.predict(X_train), color='blue')

mpl.title('Salary vs Experience (Training Set)')

mpl.xlabel('Years of Experience')

mpl.ylabel('Salary')

mpl.show()

# Visualization of the testing set result 

mpl.scatter(X_test,y_test,color='red')

mpl.plot(X_train,regressor.predict(X_train), color='blue')

mpl.title('Salary vs Experience (Training Set)')

mpl.xlabel('Years of Experience')

mpl.ylabel('Salary')

mpl.show()

