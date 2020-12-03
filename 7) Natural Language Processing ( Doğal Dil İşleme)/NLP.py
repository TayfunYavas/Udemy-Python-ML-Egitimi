# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:13:09 2020

@author: Dursun Can
"""

import numpy as np
import pandas as pd
    
yorumlar = pd.read_excel("Restaurant_Reviews.xlsx", )

import re #Regullar Expression çağırdık
import nltk

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing
derlem = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]'," ",yorumlar['Review'][i]) #Nokta vb sildik ^a-zA-Z diyerek harf haricini sildik.
    yorum = yorum.lower() #Hepsini küçük harf yaptık.
    yorum = yorum.split() #Liste haline getirdik
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] #Stopwordsleri sildi kelimelerin köklerini aldı
    yorum = ' '.join(yorum) #Aralara boşluk koyarak kelimeleri birleştirdi. Normalde ayrı kelimeler olarak görüyor.
    derlem.append(yorum)

#Feaure Extraction
#Bag Of Words(BOW)
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(max_features= 2000)

X = CV.fit_transform(derlem).toarray() #Bağımsız değişken
y = yorumlar.iloc[:,1].values  #Bağımlı değişken

#Machine Learning
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix \n",cm) 
