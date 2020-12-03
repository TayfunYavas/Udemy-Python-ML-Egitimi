# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 23:41:44 2019

@author: Dursun Can
"""

import pandas as pd
import numpy as np

data = pd.read_csv("file:///C:/Users/Dursun Can/Desktop/Veri Madenciliği/Udemy Pyhton Eğitimi/Veriler/odev_tenis.csv")

out = data.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder ()
out[:,0] = le.fit_transform(out[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder ()
out = ohe.fit_transform(out).toarray()

wind = data.iloc[:,-2:-1].values
wind[:,-2:-1] = le.fit_transform(wind[:-2:-1])

ply = data.iloc[:,-1:].values
ply[:,-1:] = le.fit_transform(ply[:-1:])

"""
s = pd.DataFrame(data =out, index=range(14), columns=["sunny","overcast","rainy"])
s1 = pd.DataFrame(data =wind, index=range(14), columns=["WindyTrue","WindyFalse"])
s2 = pd.DataFrame(data =ply, index=range(14), columns=["PlayYes","PlayNo"])

dat = data.iloc[:,1:3].values
datt = pd.DataFrame(data = dat, index=range(14), columns=["temperature","humidity"])
data1 = pd.concat([s,s1],axis=1)
data2 = pd.concat([data1,datt],axis=1)
A = pd.concat([data2,s2],axis=1)

target = A.iloc[:,5:6]
inp = A.iloc[:,0:5]
inp2 = A.iloc[:,6:10]
inputs = pd.concat([inp,inp2],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target, test_size =0.33, random_state=0)

from sklearn.linear_model import LinearRegression
rg = LinearRegression()

rg.fit(X_train,y_train)

prediction = rg.predict(X_test)

print(prediction)

import statsmodels.formula.api as sm 

BE = np.append(arr= np.ones((14,1)).astype(int),values=A, axis=1)
BE_l = A.iloc[:,[0,1,2,3,4,5,6,7,8]].values
r_ols = sm.OLS(endog = target, exog = BE_l)
r = r_ols.fit()
print(r.sumamry())
"""





