# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:19:00 2020

@author: Dursun Can
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Ads_CTR_Optimisation.csv")


import math
import random
N = 10000 #tık sayısı
d=10
toplam = 0 #toplam ödül
secilenler = []
birler = [0] * d
sifirlar = [0] * d
for n in range(1,N):
    ad = 0 #secilen ilan
    max_th = 0
    for i in range(0,d):
        rastbeta = random.betavariate (birler[i]+1, sifirlar[i]+1)
        if rastbeta > max_th:
            max_th = rastbeta
            ad = i
            
    secilenler.append(ad)
    odul = data.values[n,ad] #verilerdeki n.satır = 1 ise ödül 1
    if odul == 1:
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad] = sifirlar[ad]+1
    toplam = toplam + odul

print("Toplam Ödül:",toplam)
plt.hist(secilenler)
plt.show()