# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 23:22:24 2020

@author: Dursun Can
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('MBO.csv', header=None)

t=[] #Apriori kütüphanesi liste içinde liste istediği için bu transactionı yaptık
for i in range(0,7501):
    t.append([str(data.values[i,j]) for j in range(0,20)])

from apyori_kutuphane import apriori
rules = apriori(t,min_support=0.01,min_confidence=0.2,min_lift =3, min_length = 2)

print(list(rules))