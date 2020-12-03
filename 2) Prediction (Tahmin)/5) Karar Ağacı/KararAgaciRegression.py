# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:42:00 2020

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

#Verileri Ölçeklendirme
from sklearn.preprocessing import StandardScaler

scale1 = StandardScaler() 
x_olcekli = scale1.fit_transform(X)
scale2 = StandardScaler()
y_olcekli = scale2.fit_transform(Y)


from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,y_olcekli,color='blue')
plt.show()
print("RBF")
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')
plt.plot(x,r_dt.predict(Z), color='green')
plt.plot(x,r_dt.predict(K), color='yellow')

print(r_dt.predict([[6.6]]))
print(r_dt.predict([[6.6]]))












