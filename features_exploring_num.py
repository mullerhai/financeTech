# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 08:45:58 2017

@author: luoshichao
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

train_num=pd.read_csv('E:/geo/bank/train_data/train_num.csv')
test_num=pd.read_csv('E:/geo/bank/test_data/test_num.csv')


train_num.replace(to_replace=-999.0,value=np.nan, inplace=True)
test_num.replace(to_replace=-999.0,value=np.nan, inplace=True)

#求标准差 >0.1
train_feature_std=[train_num[item].std()  for item  in list(train_num.columns)]
num_feature_std=pd.DataFrame()
num_feature_std['num_feature_name']=list(train_num.columns)
num_feature_std['std']=train_feature_std
num_feature_std=num_feature_std.sort_values(by='std',ascending=False)

num_feature=list(num_feature_std['num_feature_name'][num_feature_std['std']>0.1])

train_num=train_num[num_feature]
test_num=test_num[num_feature]

train_num.replace(to_replace=np.nan,value=-999.0, inplace=True)
test_num.replace(to_replace=np.nan,value=-999.0, inplace=True)

#生成排序特征
train_num_rank = pd.DataFrame(train_num.mbl_num,columns=['mbl_num'])
num_feature.remove('mbl_num')
for feature in num_feature:
    train_num_rank['r'+feature] = train_num[feature].rank(method='max')
train_num_rank.to_csv('E:/geo/bank/train_data/train_num_rank.csv',index=None,encoding='utf-8')


test_num_rank = pd.DataFrame(test_num.mbl_num,columns=['mbl_num'])
for feature in num_feature:
    test_num_rank['r'+feature] = test_num[feature].rank(method='max')
test_num_rank.to_csv('E:/geo/bank/test_data/test_num_rank.csv',index=None,encoding='utf-8')


#离散特征
def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)
train_num_dis = pd.DataFrame(train_num.mbl_num,columns=['mbl_num'])
for feature in num_feature:
    train_num_dis['dis'+feature] = pct_rank_qcut(train_num[feature],10)
train_num_dis.to_csv('E:/geo/bank/train_data/train_num_dis.csv',index=None,encoding='utf-8')

test_num_dis = pd.DataFrame(test_num.mbl_num,columns=['mbl_num'])
for feature in num_feature:
    test_num_dis['dis'+feature] = pct_rank_qcut(test_num[feature],10)
test_num_dis.to_csv('E:/geo/bank/test_data/test_num_dis.csv',index=None,encoding='utf-8')


#计数特征
#import pandas as pd
#train_x = pd.read_csv('E:/geo/bank/train_data/train_num_dis.csv')

train_num_dis['n1'] = (train_num_dis==1).sum(axis=1)
train_num_dis['n2'] = (train_num_dis==2).sum(axis=1)
train_num_dis['n3'] = (train_num_dis==3).sum(axis=1)
train_num_dis['n4'] = (train_num_dis==4).sum(axis=1)
train_num_dis['n5'] = (train_num_dis==5).sum(axis=1)
train_num_dis['n6'] = (train_num_dis==6).sum(axis=1)
train_num_dis['n7'] = (train_num_dis==7).sum(axis=1)
train_num_dis['n8'] = (train_num_dis==8).sum(axis=1)
train_num_dis['n9'] = (train_num_dis==9).sum(axis=1)
train_num_dis['n10'] = (train_num_dis==10).sum(axis=1)
train_num_dis[['mbl_num','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('E:/geo/bank/train_data/train_num_nd.csv',index=None,encoding='utf-8')

test_num_dis['n1'] = (test_num_dis==1).sum(axis=1)
test_num_dis['n2'] = (test_num_dis==2).sum(axis=1)
test_num_dis['n3'] = (test_num_dis==3).sum(axis=1)
test_num_dis['n4'] = (test_num_dis==4).sum(axis=1)
test_num_dis['n5'] = (test_num_dis==5).sum(axis=1)
test_num_dis['n6'] = (test_num_dis==6).sum(axis=1)
test_num_dis['n7'] = (test_num_dis==7).sum(axis=1)
test_num_dis['n8'] = (test_num_dis==8).sum(axis=1)
test_num_dis['n9'] = (test_num_dis==9).sum(axis=1)
test_num_dis['n10'] = (test_num_dis==10).sum(axis=1)
test_num_dis[['mbl_num','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('E:/geo/bank/test_data/test_num_nd.csv',index=None,encoding='utf-8')








