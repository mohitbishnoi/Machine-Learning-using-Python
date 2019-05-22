# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:19:17 2019

@author: Mohit
"""

# =============================================================================
# SVM - Support Vector Machine
#         
#         In this algorithm, we create a hyperplan to seprate different class of target value.
#     
# Euclidean Distance - sqrt((x1-x2)**2 +.....)
# 
# Lines can be 
#             Linear
#             Non - Linear - Radial Basis Function (RBF)
# 
# 1. Kernal Type -  can be linear or (RBF)
# 2. Cost - Tuning
# 3. Gamma - it represents the angle (will only work with RBF cases)
# 
# Tuning Parameters - 
#                 For Linear - Cost
#                 RBF - Cost and gamma
# =============================================================================
                
# =============================================================================
# How To find accuracy of model
# it depends on
# 1. Confusion metrix
# 2. Accuracy Score
# 3. Classification Report
# =============================================================================

# =============================================================================
# Steps to perform Support vactor machine
# 
# 1. Extract data into python
# 2. Data cleaning if required - just remove missing value
# 3. identify target(y) and features(x)
# 4. Transform target to factors if target has numeric values(0/1)
#     if it is string values or boolean values, then factor is not needed.:
# 5. split x and y into train and test
# 6. build model using trian data
# 7. Tunne Parameter to get better accuracy (cost and gamma)
# 8. predict for test_x and validata model using test_y and pred_y
# =============================================================================

import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.head()
Y = pd.DataFrame(iris.target, columns=['species'])

x_train,x_test,y_train,y_test=train_test_split(X, Y,
                                               test_size=0.3,
                                               random_state=25)

C = 1.0 # SVM Regularization parameter
svm_model = svm.SVC(kernel='linear',C=C,
                    decision_function_shape='ovr').fit(x_train,y_train)

svm_model #here cost=1,kernel= linear
pred_y=svm_model.predict(x_test)

print(confusion_matrix(y_test,pred_y))
print(accuracy_score(y_test,pred_y))

# accuracy = 0.97

# to improve accuracy more we can tunne model
# Tunning model
# for linear model
parameter=[{'C':[1,2,5,10]}]
newmodel=GridSearchCV(svm.SVC(decision_function_shape='ovr',
                              kernel='linear'),parameter,cv=5)
#cv is cost validation - it means how many time model is tunned

newmodel.fit(x_train,y_train)
print(newmodel.best_params_)


pred_y=newmodel.predict(x_test)
print(confusion_matrix(y_test,pred_y))
print(accuracy_score(y_test,pred_y))

# for rbf model
parameter=[{'C':[1,2,5,10],'gamma':[0.1,.01,.001,.05]}]
newmodel=GridSearchCV(svm.SVC(decision_function_shape='ovr',
                              kernel='rbf'),parameter,cv=5)

newmodel.fit(x_train,y_train)
print(newmodel.best_params_)

pred_y=newmodel.predict(x_test)
print(confusion_matrix(y_test,pred_y))
print(accuracy_score(y_test,pred_y))

