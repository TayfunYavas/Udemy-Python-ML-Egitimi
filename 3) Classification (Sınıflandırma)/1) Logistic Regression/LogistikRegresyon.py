# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 00:33:32 2020

@author: Dursun Can
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("veriler.csv")
x= data.iloc[:,1:4].values
y= data.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print("Tahmin Edilen",y_pred)
print("Gerçek Deper", y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)