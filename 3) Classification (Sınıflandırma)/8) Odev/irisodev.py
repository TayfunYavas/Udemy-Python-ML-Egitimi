# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 00:33:32 2020

@author: Dursun Can
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

data = pd.read_excel("Iris.xls")
x= data.iloc[:,0:4].values
y= data.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Lojistik Regresyon

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
logr_pred = logr.predict(X_test)


#KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski") #n_neighbors 1 de 6 doğru 5 de 3 doğru buldu
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

#SVM

from sklearn.svm import SVC
svc = SVC(kernel = "rbf")
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)


#Naive Bayes

from sklearn.naive_bayes import GaussianNB #Bir çok yöntem var sklearn da bulunuyor.
gnb = GaussianNB()
gnb.fit(X_train, y_train)

gnb_pred = gnb.predict(X_test)


#Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
dt_pred = dtc.predict(X_test)


#Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)


#Değerlendirme
print("Logistic Regression Classification")
cm_logr = confusion_matrix(y_test,logr_pred)
print(cm_logr)

print("KNN Classification")
cm_knn = confusion_matrix(y_test,knn_pred)
print(cm_knn)

print("SVM Classification")
cm_svm = confusion_matrix(y_test,svc_pred)
print(cm_svm)

print("Naive Bayes Classification")
cm_nb = confusion_matrix(y_test,gnb_pred)
print(cm_nb)

print("Decision Tree Classification")
cm_dt = confusion_matrix(y_test,dt_pred)
print(cm_dt)

print("Random Forest Classification")
cm_rfc = confusion_matrix(y_test,rfc_pred)
print(cm_rfc)




















