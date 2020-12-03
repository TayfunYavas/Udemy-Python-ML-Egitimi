# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 23:41:44 2019

@author: Dursun Can
"""

import pandas as pd
import numpy as np

data = pd.read_csv("file:///C:/Users/Dursun Can/Desktop/Veri Madenciliği/Udemy Pyhton Eğitimi/Veriler/odev_tenis.csv")

from sklearn.preprocessing import LabelEncoder

data2 = data.apply(LabelEncoder().fit_transform)

windy = data2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
windy = ohe.fit_transform(windy).toarray()
print(windy)

havadurumu = pd.DataFrame(data = windy , index=range(14), columns=["overcast","rainy","sunny"])
sonveriler = pd.concat([havadurumu,data.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([data2.iloc[:,-2:],sonveriler],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
rg = LinearRegression()
rg.fit(X_train,y_train)

predictions = rg.predict(X_test)
print(predictions)

#backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values= sonveriler.iloc[:,:-1],axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog= sonveriler.iloc[:,-1:], exog=X_l)
r = r_ols.fit()
print(r.summary())

veri2 = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values= veri2.iloc[:,:-1],axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog= sonveriler.iloc[:,-1:], exog=X_l)
r = r_ols.fit()
print(r.summary())
 
"""Yeniden windy'i çıkartıp eğittik tahmin ettirdik(Windy olması sebebi sunny ve diğeri çıktı p yüksek
olduğu için windy'i de elimizle çıkardık.
"""
X_trainY = X_train.iloc[:,1:]
X_testY = X_test.iloc[:,1:]

rg.fit(X_trainY,y_train)

predictions2 = rg.predict(X_testY)
print(predictions) #ilk prediction sonucu
print(predictions2) #ikinci prediction sonucu