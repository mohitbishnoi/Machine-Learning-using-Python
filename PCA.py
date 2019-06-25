# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:10:47 2019

@author: Mohit


Principal component analysis (PCA) is to reduce the dimensionality of a data 
set consisting of many variables correlated with each other, either heavily or
lightly, while retaining the variation present in the dataset, up to the
maximum extent. The same is done by transforming the variables to a new set of
variables, which are known as the principal components (or simply, the PCs)
and are orthogonal, ordered such that the retention of variation present in 
the original variables decreases as we move down in the order. So, in this way,
the 1st principal component retains maximum variation that was present in the 
original components. The principal components are the eigenvectors of a 
covariance matrix, and hence they are orthogonal.

"""

##data - x1,x2,x3,x4,x5
##pca - pca1,pca2
##
##pca1 = w1x1+w2x2+w3x3
##pca2 = w4x4+w5x5
##
##to decide number of pca will dependent upon cumulative variance
##explaines by pca components. If cumulative variance reaches
##to .9 or above then stop forming new component.
##
##pca1 variance - 0.7      0.7
##pca2 variance - 0.23     0.93
##pca3 variance - 0.03     0.96
##pca4 variance - 0.03     0.99
##pca5 variance - 0.01     1.00

##PCA - is used to reduce variable count. whenever you many
##features then you can apply PCA to get less features


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import datasets

#Load data set
iris = datasets.load_iris()
print(iris.DESCR)
X = iris.data

#Scaling the values
X = scale(X)
#Center to the mean and component wise scale to unit variance

#Optimal number of PCA compnents should explain at least 95% of total variations
pca = PCA(n_components=4)
pca.fit(X)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print( var1)
plt.plot(range(1,5,1),var1)

#Looking at above plot I'm taking 2 PCA variables
pca = PCA(n_components=2)
pca.fit(X)
X1=pca.fit_transform(X)
print(X)
print (X1)



# PCA Assignment <  Case 1

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import datasets


data = pd.read_excel("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Python Part 2/Class 15/Assignments/Case 1/perfume_data.xlsx",heading=None)
data

data1 = data.iloc[:,1:]
data1.shape
data2 = scale(data1)

pca = PCA(n_components=10)
pca.fit(data2)
data1.corr()

# the amount of variance that each PC explains
var = pca.explained_variance_ratio_

# Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_,
                        decimals=4))

plt.plot(range(1,11,1),var1)
