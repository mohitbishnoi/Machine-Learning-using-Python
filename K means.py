# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:55:10 2019

@author: Mohit
"""

# clustering dataset
# determine k using elbow method

#clustering is used to create group of observation based on
#distance between them. One cluster is subset of given data.

#Clustering is used when you are undefine to build single
#model to with high accuracy. We divide data into cluster and
#build separate model for each cluster.

#Clustering comes under unsupervised machine learning.
#because there is no target variable.

 
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
 
plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()
 
# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
print(X)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']
 
# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
 
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#Method-2  (inertia)   help(clusters.inertia_)
cluster_range = range( 1, 10 )
cluster_errors = []

for num_clusters in cluster_range:
  clusters = KMeans( num_clusters )
  clusters.fit(X)
  cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

#plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

clusters = KMeans(3)
clusters.fit(X)
print(clusters)
print(clusters.cluster_centers_)

print(clusters.labels_)


NewX = np.array(list(zip(x1,x2,clusters.labels_))).reshape(len(x1), 3)

cluster_1=NewX[NewX[:,2]==0,:]
cluster_2=NewX[NewX[:,2]==1,:]
cluster_3=NewX[NewX[:,2]==2,:]




print(clusters.inertia_)
