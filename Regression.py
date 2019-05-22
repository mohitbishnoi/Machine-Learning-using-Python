# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:01:09 2019

@author: Mohit
"""

# =============================================================================
# Regression is used for numeric target value.
# Under this method we build the model to pridict target values based on given features
# =============================================================================

# =============================================================================
# Model validation for regression
# 1. R- Squared - higher values better equation
# 2. MSE - smaller value better equaton
# 3. RMSE - smalller value better equation
# =============================================================================

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Step 1 - Load Dataset
data = datasets.load_boston()

data.DESCR # only for builtin datasets (506 observation and  13+1 varibles)

# step 2 - Identify target and features
X = pd.DataFrame(data.data,columns=data.feature_names)

print(X)

Y = pd.DataFrame(data.target,columns=['MEDV']) # Y values (check description)
print(Y)

# Step 3 - split data into train and test

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.30,random_state=25)

# train_data = 70%  , test_data = 30%(test_size=0.30)
#random_state = (any positive number) - helps us to improve the test value at step 7

# step 4 - use train data to build model

model = LinearRegression().fit(train_x,train_y)

# Step 5 - use derived model to pridict target for test_data
pred_y = model.predict(test_x)

pred_y

# step 6 - average error in prediction
newdf = pd.DataFrame({'y':test_y['MEDV']})
newdf['pred_y']=pred_y
newdf['error']=newdf['y']-newdf['pred_y']
print(np.mean(newdf['error']))

mse = np.sum((newdf['error']-np.mean(newdf['error']))**2)/(len(newdf)-1)
mse

RMSE = np.sqrt(mse)
RMSE
# step 7 - calculate R Squared values (we are finding R squared value for test data, 
# but we can use train for comparasion purpose)

# R-squared > 0.7 - good fit model - accept model
# R-squared > 0.85 - best fit model - accept model
# R-squared < 0.5 - poor fit model - reject model

print(model.score(train_x,train_y)) # 0.7575218639671624
print(model.score(test_x,test_y)) # 0.6828624036905437

# here our train data is for good fit model,but test data is below 0.7 so we can not accept the model

# it means we need to futher improve the model



# Another method using OLS(ordinary least square) method
import statsmodels.api as sm

New_X=sm.add_constant(X)

train_x,test_x,train_y,test_y=train_test_split(New_X,Y,test_size=0.30,random_state=25)
model2 = sm.OLS(train_y,train_x).fit()
print(model2.summary())
print(model2.resquared)
pred_y = model2.predict(test_x)
error = test_y['MEDV']-pred_y
np.mean(error)
mse2 = np.sum((error-np.mean(error))**2)/(len(error)-1)
mse2
RMSE2 = np.sqrt(mse2)
RMSE2 


# comparing both Linear Regeression and OLS - 
# RMSE for Linear Regression is 4.65
# RMSE for OLS is 4.65
# Both model can be accepted

# =============================================================================
# Model validation for regression
# 1. R- Squared - higher values better equation
# 2. MSE - smaller value better equaton
# 3. RMSE - smalller value better equation
# =============================================================================

# =============================================================================
# # Steps
# 1. extract given sas file
# 2. split data into target and feature 
#     here target is "Cost" and other variables are features
# 3. check for missing values, identification variable
#     drop identification variables and replace missing value with median
# 4. split data into train and test
# 5. build model using train
# 6. use model to predict for test data
# 7. calculate error, mse
# 8. calcualated RMSE and decide accept or reject model
# =============================================================================

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Step 1 data extraction
data = pd.read_sas("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Python Part 2/Class 11/Assignment/auction.sas7bdat")
data

# step 2 - Identify target and features
data.columns
A = pd.DataFrame(data[[ 'cattle', 'calves', 'hogs', 'sheep']],columns=['cattle', 'calves', 'hogs', 'sheep'])
print(A)
B  = pd.DataFrame(data['cost'],columns=['cost'])
print(B)
 
# Step 3 check for missing value
data.isnull().sum()

# Step 4 - split data into train and test

train_a,test_a,train_b,test_b=train_test_split(A,B,test_size=0.30,random_state=25)

# train_data = 70%  , test_data = 30%(test_size=0.30)
#random_state = (any positive number) - helps us to improve the test value at step 7

# step 5 - use train data to build model

model_building = LinearRegression().fit(train_a,train_b)
print(model_building.summary())
# Step 5 - use derived model to pridict target for test_data
pred_b = model_building.predict(test_a)

pred_b

# step 6 - average error in prediction
newdf = pd.DataFrame({'b':test_b['MEDV']})
newdf['pred_b']=pred_b
newdf['error']=newdf['b']-newdf['pred_b']
print(np.mean(newdf['error']))

mse = np.sum((newdf['error']-np.mean(newdf['error']))**2)/(len(newdf)-1)
mse

RMSE = np.sqrt(mse)
RMSE
# step 7 - calculate R Squared values (we are finding R squared value for test data, 
# but we can use train for comparasion purpose)

# R-squared > 0.7 - good fit model - accept model
# R-squared > 0.85 - best fit model - accept model
# R-squared < 0.5 - poor fit model - reject model

print(model_building.score(train_a,train_b)) # 0.9072269233502167
print(model_building.score(test_a,test_b)) #0.826921107834186










