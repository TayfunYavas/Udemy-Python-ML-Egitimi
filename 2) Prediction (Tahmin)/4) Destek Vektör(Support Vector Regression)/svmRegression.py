# -*- coding: utf-8 -*-
"""
Created on Fri May 15 01:12:47 2020

@author: Dursun Can
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("file:///C:/Users/Dursun Can/Desktop/Veri Madenciliği/Udemy Pyhton Eğitimi/Veriler/maaslar.csv")

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

print("RBF")
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))
