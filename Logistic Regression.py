# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:29:07 2019

@author: Mohit
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

titanic = pd.read_csv("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Python Part 2/Class 11B/Titanic.csv")
titanic

print(titanic.head())
print(titanic.columns)
print(titanic.shape)
print(titanic.describe())

# data cleaning
# find missing values in data and replace it with median
# drop column if number of missing values are more then 40%
# in case of character column drop observation
# drop identification column such as name,ID etc

# you will get 898,8 dataframe after dropping
titanic.isnull().sum() # age have 177 missing value
(titanic.isnull().sum()/len(titanic))*100 # to check percentage of missing values in that column

# de median and store in same age column
titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())

# dropping column cabin as it have more then  40% missing value 
titanic=titanic.drop(['Cabin'],axis=1)
# dropping missing observation from embarked
titanic=titanic.dropna(axis=0)
# again check for missing value
titanic.isnull().sum()
# dropping identification variable
titanic = titanic.drop(['PassengerId','Name','Ticket'],axis=1)
titanic.shape # we got dataframe as 889,8


Y = titanic.iloc[:,0]
X = titanic.iloc[:,1:]

# convert character variables to dummy variables

X.head()

# sex and embarked are charater variables

gender = pd.get_dummies(titanic['Sex'],drop_first=True)
gender.head()

Embark = pd.get_dummies(titanic['Embarked'],drop_first=True)
Embark.head()

# combine dummy variable to dataset and drop catergoical variable

titanic_data3 = titanic.drop(['Sex','Embarked'],1)

titanic_data4 = pd.concat([titanic_data3,gender,Embark],axis=1)

Y = titanic_data4.iloc[:,0]
X = titanic_data4.iloc[:,1:]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=.30,random_state=10)

logistic_model=LogisticRegression().fit(train_x,train_y)

print(logistic_model.coef_) #weight value
print(logistic_model.intercept_) #bias value

pred_y=logistic_model.predict(test_x)

print(confusion_matrix(test_y,pred_y))

print(accuracy_score(test_y,pred_y))


# Case study 2

adult = pd.read_csv("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Python Part 2/Class 11B/Logistic regression/Case2/adult.csv",na_values='?')
adult

(adult.isnull().sum()/len(adult))*100 # to check percentage of missing values in that column

adult.columns

data2 = adult[['age','fnlwgt', 'educational-num', 'gender',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']]

data2A = data2.dropna()
data2A.shape #(47985, 9)

Income = list(map(lambda x:1 if(x==">50K") else 0,data2A['income']))

data2A['Y']=Income
data2A=data2A.drop(['income'],axis=1)

gender = pd.get_dummies(data2A['gender'],drop_first=True)
gender.head()

country = pd.get_dummies(data2A['native-country'],drop_first=True)
country.head()

# as we are getting 41 unique country name, so we are dropping this.

# combine dummy variable to dataset and drop catergorical variable

data2B=data2A.drop(['gender','native-country'],1)
data3=pd.concat([data2B,gender],axis=1)

data3.shape
data3.columns

X = data3.drop(['Y'],axis=1)
Y = data3['Y']

X.shape
X.head()
Y.head()


train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=.4,random_state=10)

model=LogisticRegression().fit(train_x,train_y)

pred_y=model.predict(test_x)

print(confusion_matrix(test_y,pred_y))

print(accuracy_score(test_y,pred_y))

#accuracy of model is 0.79 - good fit model




