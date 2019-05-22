# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:25:21 2019

@author: Mohit
"""
# =============================================================================
# # 1. Missing Values
# # 2. Outlier Detection
# =============================================================================


import numpy as np
import pandas as pd

data = pd.DataFrame(np.random.randn(10,4),columns=list('ABCD'))

print(data)

# How to find missing values in data
# Gives count of missing values for each variables
data.isnull().sum() # most comonly used

#gives count of missing values for given variable
np.sum(np.isnan(data['A']))

# How to remove missing values
# dropping all observation which has missing values
data1=data
data1.dropna()

#dropping all variables which has missing values
data2=data
data2_Nonmissing=data2.dropna(axis=1)
data2_Nonmissing

#Missing values treatment
# 1. replace missing value with previous values
# replace missing value with 0

data3=data
data3.replace(np.nan,0)

#replace missing with previous values
data3.fillna(method='pad') #fill forward

#replace missing values with next value
data3.fillna(method='bfill') #fill backward


# replace missing values with median
data3.fillna(data3.median()) #most comonly  used
 #column A median will replace column A missing value respectivelly
 
 
 
# =============================================================================
#  #use case
# =============================================================================
import os
os.chdir("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Python Part 2/Class 9")
os.getcwd()
os.listdir()
 
dataset=pd.read_table('pima-indians-diabetes.data.txt',sep=",",skiprows=12,header=None)
print(dataset.describe())

#mark zero values as misssing or NaN
dataset[[1,2,3,4,5]]=dataset[[1,2,3,4,5]].replace(0,np.NaN)

#count the number of NaN values in each column
print(dataset.isnull().sum())
dataset.shape

#perfrom folowing on given dataset
# 1. variable 1 - replace missing values with given value
# 2. variable 2 - replace missing value with 40
# 3. variable 3 - replace missing value with median
# 4. varibale 4 - drop the variable
# 5. variable 5 - drop missing value observation

dataset[1]=dataset[1].replace(method='pad')
dataset[2]=dataset[2].replace(np.nan,40)
dataset[3]=dataset[3].fillna(dataset[3].median())
dataset=dataset.drop([4],axis=1)
dataset=dataset.dropna(axis=0)

# =============================================================================
# # outlier values - detection
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(10)
# seed - is used to re-produce same result again and again.
# whenever we use a random function to genrate random value, every time
# we get new values, that can genrate different results.
# in case you want same result every time, then set seed to any positve number value
# integer value

array1 = np.random.normal(100,10,200)
array2 = np.random.normal(90,20,200)
 
data = [array1,array2]
data
len(data[0])

plt.boxplot(data)
res = plt.boxplot(data)

array1_outlier = res["fliers"][0].get_data()[1]
array1_outlier
#outlier value are
# array([ 70.20403229,  77.04896671, 123.94703665, 124.67651056,
      # 124.04325606, 124.65325082])

array2_outlier = res["fliers"][1].get_data()[1]
array2_outlier
# outlier value are
# array([143.59820616])

res['boxes'][0].get_data()[1]

# Method 2  - when we have huge (big data = millions of observation)
# formula - mean+3*std - values above or below this formula are taken as outlier
df = pd.DataFrame(array1,columns={'Data'})
df.describe()

df[(np.abs(df.Data-df.Data.mean())>(3*df.Data.std()))]
# outlier value are
# 70.24032










