# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 02:03:37 2019

@author: Dursun Can
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("maaslar.csv")

x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values

#Linear Regresyon
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X), color="blue")

#Polinomal Regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()


#Tahminler
print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))