# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 22:29:45 2020

@author: Dursun Can
"""

import pandas as pd

data = pd.read_csv("satislar.csv")

X = data.iloc[:,0:1].values
Y = data.iloc[:,1].values
bolme = 0.33

from sklearn import model_selection
X_train , X_test, Y_train,Y_test = model_selection.train_test_split(X, Y,test_size = bolme)
"""
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

print(lr.predict(X_test))
"""
import pickle #Gereken kütüphane
dosya = "model.kayit"
pickle.dump(lr,open(dosya,'wb')) #Kaydetmek için


yuklenen = pickle.load(open(dosya,'rb')) #Yüklemek için
print(yuklenen.predict(X_test))










