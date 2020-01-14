# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 11:15:52 2017

@author: luoshichao
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 08:45:58 2017

@author: luoshichao
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

train_int=pd.read_csv('E:/geo/bank/train_data/train_int.csv')
test_int=pd.read_csv('E:/geo/bank/test_data/test_int.csv')

train_int.replace(to_replace=-999.0,value=np.nan, inplace=True)
test_int.replace(to_replace=-999.0,value=np.nan, inplace=True)

#求标准差 >0.1
train_feature_std=[train_int[item].std()  for item  in list(train_int.columns)]
int_feature_std=pd.DataFrame()
int_feature_std['int_feature_name']=list(train_int.columns)
int_feature_std['std']=train_feature_std
int_feature_std=int_feature_std.sort_values(by='std',ascending=False)

int_feature=list(int_feature_std['int_feature_name'][int_feature_std['std']>0.1])

train_int=train_int[int_feature]
train_int.replace(to_replace=np.nan,value=-999.0, inplace=True)

test_int=test_int[int_feature]
test_int.replace(to_replace=np.nan,value=-999.0, inplace=True)


#生成排序特征
train_int_rank = pd.DataFrame(train_int.mbl_num,columns=['mbl_num'])
int_feature.remove('mbl_num')
for feature in int_feature:
    train_int_rank['r'+feature] = train_int[feature].rank(method='max')/float(len(train_int[feature]))
    
train_int_in=pd.merge(train_int,train_int_rank,on='mbl_num')
train_int_in.to_csv('E:/geo/bank/train_data/train_int_in.csv',index=None,encoding='utf-8')

#生成排序特征
test_int_rank = pd.DataFrame(test_int.mbl_num,columns=['mbl_num'])
for feature in int_feature:
    test_int_rank['r'+feature] = test_int[feature].rank(method='max')/float(len(test_int[feature]))
    
test_int_in=pd.merge(test_int,test_int_rank,on='mbl_num')
test_int_in.to_csv('E:/geo/bank/test_data/test_int_in.csv',index=None,encoding='utf-8')











