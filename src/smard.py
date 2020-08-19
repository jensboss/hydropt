#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:46:51 2020

@author: Jens
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

path = 'data'
files = [os.path.join(path, fname) for fname in os.listdir(path) 
         if not fname.startswith('data')]


df = pd.DataFrame()
for file in files:
    df_temp = pd.read_csv(file, sep=';', parse_dates={'time':[0,1]})
    df_temp.set_index('time', inplace=True)
    df = pd.concat((df,df_temp))
    
    
    
df = df.tz_localize('Europe/Zurich', ambiguous='infer')
df = df.tz_convert(None)

df = df.sort_index()
    
plt.figure(0)
plt.clf()
df.plot(ax=plt.gca())


df.to_csv(os.path.join(path, 'data_2019.csv'), sep=';')