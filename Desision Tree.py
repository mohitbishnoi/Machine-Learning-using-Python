# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:23:58 2019

@author: Mohit
"""

# Tree Model - In tree model method. we get a tree structure to decide target value
# tree will be working on fillowing algorithm 

# 1. Cart Model (Classification and Regression Tree)
#           -- Gini Index (Sigma(E) P^^2+Q^^2)

# 2. ID3 (Iterative Dichotomiser 3)
#           -- (Entropy          &            Indormative Gain)
#           -- (Sigma(E) Plog2P)         (Entropy of target - Entropy of feature)
            # P is Probablity of success
            # Q is P-1


# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.
 
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
 
# Function importing Dataset
def importdata():
    balance_data = pd.read_csv("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Python Part 2/Class 12/balance_scale.csv",
    sep= ',', header = None)
     
    # Printing the dataswet shape
    print ("Dataset Lenght: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)
     
    # Printing the dataset obseravtions
    print ("Dataset: ",balance_data.head(6))
    return balance_data
 
# Function to split the dataset
def splitdataset(balance_data):
 
    # Seperating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]
    
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
     
    return X, Y, X_train, X_test, y_train, y_test
     
# Function to perform training with giniIndex.
def train_using_gini(X_train, y_train):
 
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
 
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
     
# Function to perform training with entropy.
def tarin_using_entropy(X_train, y_train):
 
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
 
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
 
 
# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
     
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("Report : ",
    classification_report(y_test, y_pred))
 
# Driver code
def main():
     
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = tarin_using_entropy(X_train,  y_train)
     
    # Operational Phase
    print("Results Using Gini Index:")
     
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    print("prediction result  :", y_pred_gini)
    cal_accuracy(y_test, y_pred_gini)
     
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
     
     
# Calling main function
if __name__=="__main__":
    main()
