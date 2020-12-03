# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:40:43 2019

@author: Dursun Can
"""

import pandas as pd
import numpy as np

data = pd.read_csv("file:///C:/Users/Dursun Can/Desktop/Veri Madenciliği/Udemy Pyhton Eğitimi/Veriler/veriler.csv")

#Kategorikten nümerik veriye
ulke =  data.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
ulke = ohe.fit_transform(ulke).toarray()


cins =  data.iloc[:,-1:].values
print(cins)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cins[:,0] = le.fit_transform(cins[:,0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
cins = ohe.fit_transform(cins).toarray()
print(cins)

sonuc = pd.DataFrame(data = ulke, index=range(22), columns=["fr","tr","us"])

sonuc2 = pd.DataFrame(data = Yas, index=range(22), columns=["boy","kilo","yas"] )

cinsiyet = data.iloc[:,-1].values


sonuc3 = pd.DataFrame(data = cins[:,:1], index=range(22), columns=["cinsiyet"])

s= pd.concat([sonuc,sonuc2],axis=1)

s2= pd.concat([s,sonuc3],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
    
boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

X_train, X_test, y_train, y_test = train_test_split(veri,boy,test_size=0.33, random_state = 0)

r2 = LinearRegression()

r2.fit(X_train,y_train)

y_predict = r2.predict(X_test)


import statsmodels.formula.api as sm #Backward Elimination için kullanılır P değeri en yüksek olanlar çıkartılır

X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary()) #Sonuca göre x5 çıkar yani 4. indis

X_l = veri.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary()) #Böyle bırakılabilir ama eğer yeterli görülmezse x5 yani 5.indis çıkar

X_l = veri.iloc[:,[0,1,2,3]].values
r_ols = sm.OLS(endog = boy, exog = X_l)
r = r_ols.fit()
print(r.summary()) #Bütün p değerleri 0



