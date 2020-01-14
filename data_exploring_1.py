# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:30:53 2017

@author: luoshichao
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

sample_label=pd.read_csv('E:/geo/bank/sample_label.dst',sep='\t')
sample_bak=pd.read_csv('E:/geo/bank/sample.bak',sep='\t')
sample_bak=sample_bak[sample_bak['mbl_num'].notnull()]
sample_bak['mbl_num']=[str(int(item)) for item in sample_bak['mbl_num']]
sample_label['mbl_num']=[str(int(item)) for item in sample_label['mbl_num']]
data=pd.merge(sample_label,sample_bak,on='mbl_num')
data['target']=0
data['target'][data['overdue_days']>30]=1
data=data[['mbl_num','target','iden_num_x','loan_dt','bankcard_num_x']]


feature_data=pd.read_csv('E:/geo/bank/data/feature_data.csv',low_memory=False)

train=pd.merge(data,feature_data,left_on='bankcard_num_x',right_on='card')

del data,feature_data,sample_bak,sample_label

#删除已知明显的重复字段
train=train.drop(['bankcard_num_x','card','time','年月'],axis=1)
train=train.drop_duplicates()
col_name=list(train.columns)
col_name.remove('target')
#train.to_csv('E:/geo/bank/data/train.csv',index=None)
test_all_data=pd.read_csv('E:/geo/test_all_data.csv',low_memory=False)
test=test_all_data[col_name]
test=test.drop_duplicates()



#缺少值处理
train.fillna(-999.0,inplace=True)
test.fillna(-999.0,inplace=True)
#train.replace(to_replace=-9999999,value=-1, inplace=True)

#分别对列和行做统计
train_feature_miss=(train==-999.0).sum(axis=0).reset_index()
train_feature_miss.columns=['feature','null_num']
train_feature_miss=train_feature_miss.sort_values(by='null_num',ascending=False)
train_feature_miss['null_odds']=train_feature_miss['null_num']/train_feature_miss['null_num'].max()


train_feature_miss=(train==-999.0).sum(axis=0).reset_index()
train_feature_miss.columns=['feature','null_num']
train_feature_miss=train_feature_miss.sort_values(by='null_num',ascending=False)
train_feature_miss['null_odds']=train_feature_miss['null_num']/train_feature_miss['null_num'].max()

'''
test_feature_miss=(test==-999.0).sum(axis=0).reset_index()
test_feature_miss.columns=['feature','null_num']
test_feature_miss=test_feature_miss.sort_values(by='null_num',ascending=False)
test_feature_miss['null_odds']=test_feature_miss['null_num']/test_feature_miss['null_num'].max()
'''

#有时间可以画个图

feature=list(train_feature_miss['feature'][train_feature_miss['null_odds']<0.7])



train['n_null']=(train==-999.0).sum(axis=1)
train_1 = train[['mbl_num','target','n_null']]
train_1=train_1.sort_values(by='n_null')
t_1=train_1['n_null'].values
y_1=train_1.target.cumsum()
x=range(len(t_1))
plt.scatter(x,t_1,c='k')
plt.plot(x,y_1,c='b')
plt.title('train set')
plt.show()


test['n_null']=(test==-999.0).sum(axis=1)
test_1 = test[['mbl_num','n_null']]
test_1=test_1.sort_values(by='n_null')
t_1=test_1['n_null'].values
x=range(len(t_1))
plt.scatter(x,t_1,c='k')
plt.title('test set')
plt.show()





train_1 = train_1[train_1['n_null']<100]
test_1 = test_1[test_1['n_null']<100]
#train_1.to_csv('E:/geo/bank/data/train_1.csv',index=None,encoding='utf-8')

train_1['discret_null'] = train_1.n_null
train_1.discret_null[train_1.discret_null<=20] = 1
train_1.discret_null[(train_1.discret_null>20)&(train_1.discret_null<=40)] = 2
train_1.discret_null[(train_1.discret_null>40)&(train_1.discret_null<=50)] = 3
train_1.discret_null[(train_1.discret_null>50)] = 4
train_1=train_1[['mbl_num','discret_null']]
train_1.to_csv('E:/geo/bank/data/train_n_null.csv',index=None,encoding='utf-8')


test_1['discret_null'] = test_1.n_null
test_1.discret_null[test_1.discret_null<=20] = 1
test_1.discret_null[(test_1.discret_null>20)&(test_1.discret_null<=40)] = 2
test_1.discret_null[(test_1.discret_null>40)&(test_1.discret_null<=50)] = 3
test_1.discret_null[(test_1.discret_null>50)] = 4
test_1=test_1[['mbl_num','discret_null']]
test_1.to_csv('E:/geo/bank/data/test_n_null.csv',index=None,encoding='utf-8')



data_1=train[feature]
data_2=pd.DataFrame(train_1['mbl_num'])
train_data=pd.merge(data_1,data_2,on='mbl_num')
del data_1,data_2
train_data.to_csv('E:/geo/bank/data/train_data.csv',index=None,encoding='utf-8')

feature.remove('target')
data_11=test[feature]
data_22=pd.DataFrame(test_1['mbl_num'])
test_data=pd.merge(data_11,data_22,on='mbl_num')

test_data.to_csv('E:/geo/bank/data/test_data.csv',index=None,encoding='utf-8')