# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:08:41 2019

@author: Dursun Can
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("satislar.csv")
aylar = data[["Aylar"]]
satislar = data[["Satislar"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(aylar,satislar,test_size = 0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

#Veriler sıralı değil random olduğu için.Bunları indexe göre sıralıyoruz.Yoksa Grafik doğru olmaz
X_train = X_train.sort_index()
y_train = y_train.sort_index()

#Görselleştirme için kullanıyor
plt.plot(X_train,y_train)
plt.plot(X_test,predictions)

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")

