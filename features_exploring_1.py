# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 06:29:12 2017

@author: luoshichao
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

train=pd.read_csv('E:/geo/bank/data/train_data.csv')
test=pd.read_csv('E:/geo/bank/data/test_data.csv')

all_features=pd.read_csv('E:/geo/bank/data/all_features.csv')[['变量名称','字段类型']]

train_features=list(train.columns)
features_in=[item for item in train_features if item in  list(all_features['变量名称'])]
features_out=[item for item in train_features if item not in list(all_features['变量名称'])]

train_data_out=train[features_out]
train_data_out.to_csv('E:/geo/bank/train_data/train_data_out.csv',index=None,encoding='utf-8')

features_out.remove('target')
test_data_out=test[features_out]
test_data_out.to_csv('E:/geo/bank/test_data/test_data_out.csv',index=None,encoding='utf-8')

features=pd.DataFrame(features_in,columns=['features_in'])

features=pd.merge(features,all_features,left_on='features_in',right_on='变量名称')
features_in=features[['变量名称','字段类型']]
del all_features,train_features
features_in.to_csv('E:/geo/bank/data/features_in.csv',index=None,encoding='utf-8')
dict={'decimal(19,2)':'numberic',
      'decimal(19,3)':'numberic', 
      'decimal(19,4)':'numberic',
             'bigint':'numberic',
             'double':'numberic'}
features_in['字段类型']=features_in['字段类型'].map(lambda x:dict.get(x,x))
features_in.to_csv('E:/geo/bank/data/features_type.csv',index=None,encoding='utf-8')

num_feature=list(features_in['变量名称'][features_in['字段类型']=='numberic'])
int_feature=list(features_in['变量名称'][features_in['字段类型']=='int'])
string_feature=list(features_in['变量名称'][features_in['字段类型']=='string'])

train_num=train[['mbl_num']+num_feature]
train_int=train[['mbl_num']+int_feature]
train_string=train[['mbl_num']+string_feature]
train_num.to_csv('E:/geo/bank/train_data/train_num.csv',index=None,encoding='utf-8')
train_int.to_csv('E:/geo/bank/train_data/train_int.csv',index=None,encoding='utf-8')
train_string.to_csv('E:/geo/bank/train_data/train_string.csv',index=None,encoding='utf-8')

test_num=test[['mbl_num']+num_feature]
test_int=test[['mbl_num']+int_feature]
test_string=test[['mbl_num']+string_feature]
test_num.to_csv('E:/geo/bank/test_data/test_num.csv',index=None,encoding='utf-8')
test_int.to_csv('E:/geo/bank/test_data/test_int.csv',index=None,encoding='utf-8')
test_string.to_csv('E:/geo/bank/test_data/test_string.csv',index=None,encoding='utf-8')





