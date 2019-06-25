# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:23:54 2019

@author: Mohit
"""

# Topic - KNN Model


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score  

iris_data = datasets.load_iris()
iris_data.keys()

X = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
Y = pd.DataFrame(iris_data.target,columns=['spicies'])

X.isnull().sum()
Y.isnull().sum()

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)

KNN_model = KNeighborsClassifier(n_neighbors=5).fit(train_x,train_y)

pred_y=KNN_model.predict(test_x)

print("Confusion Matrix:",confusion_matrix(test_y,pred_y))
print("Accuracy_score:",accuracy_score(test_y,pred_y))
print("Classificaton Report:",classification_report(test_y,pred_y))


error=np.mean(test_y['spicies']!=pred_y)
print(error)

# 0.02 - error value at k=5
# 0.966 - accuracy score at k = 5

# run a for loop for various value at k and find error at every k value
# we will select k which has minimum error

error=[]
all_k=[1,3,5,7,9,10,15,20]
for i in all_k:
    KNN_model=KNeighborsClassifier(n_neighbors=i).fit(train_x,train_y)
    pred_y=KNN_model.predict(test_x)
    error.append(np.mean(test_y['spicies']!=pred_y))

print(error)
plt.plot(range(1,9),error)

#final model
KNN_model = KNeighborsClassifier(n_neighbors=9).fit(train_x,train_y)

pred_y=KNN_model.predict(test_x)

print("Accuracy_score:",accuracy_score(test_y,pred_y))

