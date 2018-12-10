# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:55:18 2018

@author: 1628065
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ccpp.csv')
X= dataset.iloc[:,1:2].values
y=dataset.iloc[:,4].values
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)   # 
#predicting a regressor model
y_pred=regressor.predict(X_test)

#plotting the testing set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Simple Linear Regression')
plt.xlabel('at->')
plt.ylabel('pe->')

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

from sklearn.metrics import accuracy_score
percentage_accuracy= accuracy_score(Y_test,y_pred)*100
