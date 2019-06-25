# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:26:59 2019

@author: Mohit
"""

# Topic : GBM - gradient boosting model

# Multiple tree are created sequentially
# Can be used when we have weak feature

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor 

boston_data = datasets.load_boston()

boston_data.keys()

X = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
Y = pd.DataFrame(boston_data.target,columns=['MEDV'])

X.head()
Y.head()

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)

GBM_model =GradientBoostingRegressor().fit(train_x,train_y)

pred_y=GBM_model.predict(test_x)

mean_squared_error(test_y,pred_y)
#average error in predicting target values

GBM_model.score(train_x,train_y)
# R-squared = 0.98 - this means model is best fit model


# Classification

email_spam=pd.read_excel("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Python Part 2/Class 17/email_spam.xlsx")

email_spam.head()

email_spam.columns

#(['crl.tot', 'dollar', 'bang', 'money', 'n000', 'make', 'spam'], dtype='object')

# target - spam

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

X = pd.DataFrame(email_spam[['crl.tot', 'dollar', 'bang', 'money', 'n000', 'make']],columns=['crl.tot', 'dollar', 'bang', 'money', 'n000', 'make'])
Y = pd.DataFrame(email_spam['spam'],columns=['spam'])

X.head()
Y.head()

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)

GBM_classifier = GradientBoostingClassifier().fit(train_x,train_y)

pred_y=GBM_classifier.predict(test_x)

print(classification_report(test_y,pred_y))
print(accuracy_score(test_y,pred_y))

#accuracy score = 0.87, it means best fit model


