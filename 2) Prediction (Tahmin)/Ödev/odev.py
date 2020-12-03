# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:54:53 2020

@author: Dursun Can
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm


data = pd.read_csv("maaslar_yeni.csv")

x = data.iloc[:,2:5]
y = data.iloc[:,5:]
X = x.values
Y = y.values

print(data.corr())

#Algoritmalar

#Lin_Reg
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

model = sm.OLS(lr.predict(X),X)
print(model.fit().summary())

print("Linear R2 degeri:")
print(r2_score(Y, lr.predict((X))))

#Pol_Reg
from sklearn.preprocessing import PolynomialFeatures
pl_reg = PolynomialFeatures(degree = 4)
x_poly = pl_reg.fit_transform(X)
print(x_poly)
lr2 = LinearRegression()
lr2.fit(x_poly,y)

print('poly ols')
model2 = sm.OLS(lr2.predict(pl_reg.fit_transform(X)),X)
print(model2.fit().summary())

print("Polynomial R2 degeri:")
print(r2_score(Y, lr2.predict(pl_reg.fit_transform(X)) ))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)


print('svr ols')
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )


#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)



print('dt ols')
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)) )



#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X,Y)




print('dt ols')
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)) )


#ozett R2 degerleri
print('----------------')
print("Linear R2 degeri:")
print(r2_score(Y, lr.predict((X))))


print("Polynomial R2 degeri:")
print(r2_score(Y, lr2.predict(pl_reg.fit_transform(X)) ))


print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )


print("Decision Tree R2 degeri:")
print(r2_score(Y, r_dt.predict(X)) )

print("Random Forest R2 degeri:")
print(r2_score(Y, rf_reg.predict(X)) )


#Sonuçları