# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:49:12 2020

@author: Dursun Can
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("wine.csv")

X = data.iloc[:,1:13].values
y = data.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#Pcaden önce  gelen LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)

#pca sonra gelen LR
lr2 = LogisticRegression(random_state=0)
lr2.fit(X_train2,y_train)

#tahminler
y_pred = lr.predict(X_test) #Pca uygulanmamış

y_pred2 = lr2.predict(X_test2) #Pca uygulanmış

from sklearn.metrics import confusion_matrix
#actual  vs pca olmadan
print("Gerçek vs Pcasiz")
cm = confusion_matrix(y_test,y_pred)
print(cm)

#actual vs pca sonrası
print("Gerçek vs pca")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

#pca sonrası vs pca sonrası
print("pcasiz vs pca")
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)
