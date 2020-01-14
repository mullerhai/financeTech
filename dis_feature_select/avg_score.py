# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:22:39 2017

@author: luoshichao
"""

#coding=utf-8

import pandas as pd 
import os


files = os.listdir('E:/geo/code/dis_feature_select/feature_score')
fs = {}
for f in files:
    t = pd.read_csv('E:/geo/code/dis_feature_select/feature_score/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

with open('dis_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)




