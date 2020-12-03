# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:19:00 2020

@author: Dursun Can
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Ads_CTR_Optimisation.csv")

#Random Selection
"""
import random 

N = 10000
d = 10
toplam = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = data.values[n,ad] #verilerdeki n.satır = 1 ise ödül 1
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()
"""
import math
#UCB
N = 10000 #tık sayısı
d=10
oduller = [0] * d #ik başta bütün ilanların ödülü 0
toplam = 0 #toplam ödül
tiklamalar = [0] * d #o ana kadar ki tıklamalar
secilenler = []
for n in range(1,N):
    ad = 0 #secilen ilan
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i] > 0):
            ortalama = oduller[i]/tiklamalar[i]
            delta = math.sqrt(3/2* math.log(n)/tiklamalar[i])
            ucb = ortalama+delta
        else:
            ucb = N*10
        if max_ucb < ucb: #maxtan büyük ucb çıktı güncelle
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+1
    odul = data.values[n,ad] #verilerdeki n.satır = 1 ise ödül 1
    oduller[ad] = oduller[ad] + odul
    toplam = toplam + odul

print("Toplam Ödül:",toplam)
plt.hist(secilenler)
plt.show()