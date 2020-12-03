# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:20:14 2020

@author: Dursun Can
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('maaslar.csv')
x = data.iloc[:,1:2]
y = data.iloc[:,2:]
X = x.values
Y = y.values
Z = X + 0.5
K = X - 0.4
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators=10)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y, color='red')
plt.plot(x,rf_reg.predict(X),color='blue')
plt.plot(x,rf_reg.predict(Z),color='green')
plt.plot(x,rf_reg.predict(K),color='yellow')
