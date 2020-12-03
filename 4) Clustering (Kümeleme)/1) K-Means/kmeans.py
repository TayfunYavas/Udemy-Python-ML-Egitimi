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