# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:58:41 2020

@author: Dursun Can
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("musteriler.csv")

X = data.iloc[:,3:].values

#K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_) #Noktaları verir bize
sonuclar = []
for i in range(1,11):
    kmeansdeneme = KMeans (n_clusters = i, init="k-means++",random_state=123)
    kmeansdeneme.fit(X)
    sonuclar.append(kmeansdeneme.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()
kmeansdeneme = KMeans (n_clusters = 4, init="k-means++",random_state=123)
y_predict= kmeansdeneme.fit_predict(X)
print(y_predict)
plt.title("K-Means")
plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s = 100, c="red")
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s = 100, c="blue")
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s = 100, c="green")
plt.scatter(X[y_predict==3,0],X[y_predict==3,1],s = 100, c="yellow")
plt.show()
#HC

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
y_predict = ac.fit_predict(X)
print(y_predict)

plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s=100, c="red")
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s=100, c="blue")
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s=100, c="green")
plt.scatter(X[y_predict==3,0],X[y_predict==3,1],s = 100, c="yellow")
plt.title("HC")
plt.show()

import scipy.cluster.hierarchy as sch #Dendogram için import ettik scikitlearn gibi kütüphane

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.show()