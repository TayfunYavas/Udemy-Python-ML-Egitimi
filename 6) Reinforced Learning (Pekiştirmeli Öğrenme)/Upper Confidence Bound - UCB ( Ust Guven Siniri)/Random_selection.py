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