# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:55:08 2017

@author: luoshichao
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

features_dis = list(pd.read_csv('E:/geo/code/dis_feature_select/dis_feature_score.csv')['feature'])
features_rank = list(pd.read_csv('E:/geo/code/rank_feature_select/rank_feature_score.csv')['feature'])
features_or = list(pd.read_csv('E:/geo/code/ori_feature_select/ori_feature_score.csv')['feature'])

#data_out_processing
train_data_out=pd.read_csv('E:/geo/bank/train_data/train_data_out.csv')
train_data_out['age']=[2017-int(item[6:10]) for item in train_data_out['iden_num_x']]
train_data_out['loan_dt'] = pd.to_datetime(train_data_out['loan_dt'],dayfirst=True)
train_data_out['loan_weekday'] = train_data_out['loan_dt'].apply(lambda x:x.weekday()+1)
train_data_out=train_data_out.drop(['iden_num_x','loan_dt'],axis=1)
train_data_out=train_data_out[['mbl_num','target','age', 'loan_weekday']]

test_data_out=pd.read_csv('E:/geo/bank/test_data/test_data_out.csv')
def to_age(x):
    x=str(x)
    if len(x)==18:
        return 2017-int(x[6:10])
    else:
        return np.nan
test_data_out['age']=test_data_out['iden_num_x'].apply(to_age)
test_data_out['age'].fillna(int(test_data_out['age'].mean()),inplace=True)
test_data_out['loan_dt'] = pd.to_datetime(test_data_out['loan_dt'],dayfirst=True)
test_data_out['loan_weekday'] = test_data_out['loan_dt'].apply(lambda x:x.weekday()+1)
test_data_out=test_data_out.drop(['iden_num_x','loan_dt'],axis=1)
test_data_out=test_data_out[['mbl_num','age', 'loan_weekday']]

#ori_data_processing
train_num=pd.read_csv('E:/geo/bank/train_data/train_num.csv')
train_int=pd.read_csv('E:/geo/bank/train_data/train_int_in.csv')
train_string=pd.read_csv('E:/geo/bank/train_data/train_string_nan.csv')
train_original=pd.merge(train_data_out,train_num,on='mbl_num')
train_original=pd.merge(train_original,train_int,on='mbl_num')
train_original=pd.merge(train_original,train_string,on='mbl_num')
del train_num,train_int,train_string

test_num=pd.read_csv('E:/geo/bank/test_data/test_num.csv')
test_int=pd.read_csv('E:/geo/bank/test_data/test_int_in.csv')
test_string=pd.read_csv('E:/geo/bank/test_data/test_string_nan.csv')
test_original=pd.merge(test_data_out,test_num,on='mbl_num')
test_original=pd.merge(test_original,test_int,on='mbl_num')
test_original=pd.merge(test_original,test_string,on='mbl_num')
del test_num,test_int,test_string,train_data_out,test_data_out

train_colname=list(train_original.columns)
train_colname.remove('target')
test_original=test_original[train_colname]


col_names_train=['x_ori_'+str(item) for item in range(2,len(train_original.columns))]
train_original.columns=['mbl_num','target']+col_names_train
test_original.columns=['mbl_num']+col_names_train

train_original_data=train_original[['mbl_num','target']+features_or[:500]]
test_original_data=test_original[['mbl_num']+features_or[:500]]

#rank_data_processing

train_num_rank=pd.read_csv('E:/geo/bank/train_data/train_num_rank.csv')
test_num_rank=pd.read_csv('E:/geo/bank/test_data/test_num_rank.csv')

train_colname=list(train_num_rank.columns)
test_num_rank=test_num_rank[train_colname]

col_names_rank=['x_rank_'+str(item) for item in range(1,len(train_num_rank.columns))]
train_num_rank.columns=['mbl_num']+col_names_rank
test_num_rank.columns=['mbl_num']+col_names_rank

train_num_rank_data=train_num_rank[['mbl_num']+features_rank[:400]]
test_num_rank_data=test_num_rank[['mbl_num']+features_rank[:400]]

#dis_data_processing
train_num_dis=pd.read_csv('E:/geo/bank/train_data/train_num_dis.csv')
test_num_dis=pd.read_csv('E:/geo/bank/test_data/test_num_dis.csv')

train_colname=list(train_num_dis.columns)
test_num_dis=test_num_dis[train_colname]

col_names_dis=['x_dis_'+str(item) for item in range(1,len(train_num_dis.columns))]
train_num_dis.columns=['mbl_num']+col_names_dis
test_num_dis.columns=['mbl_num']+col_names_dis

train_num_dis_data=train_num_dis[['mbl_num']+features_dis[:400]]
test_num_dis_data=test_num_dis[['mbl_num']+features_dis[:400]]

#other 11 features
train_num_nd=pd.read_csv('E:/geo/bank/train_data/train_num_nd.csv')
train_n_null=pd.read_csv('E:/geo/bank/data/train_n_null.csv')
train_num_nd=pd.merge(train_num_nd,train_n_null,on='mbl_num')
del train_n_null

test_num_nd=pd.read_csv('E:/geo/bank/test_data/test_num_nd.csv')
test_n_null=pd.read_csv('E:/geo/bank/data/test_n_null.csv')
test_num_nd=pd.merge(test_num_nd,test_n_null,on='mbl_num')
del test_n_null


train =pd.merge(train_original_data,train_num_rank_data,on='mbl_num')
train =pd.merge(train,train_num_dis_data,on='mbl_num')
train =pd.merge(train,train_num_nd,on='mbl_num')

test =pd.merge(test_original_data,test_num_rank_data,on='mbl_num')
test =pd.merge(test,test_num_dis_data,on='mbl_num')
test =pd.merge(test,test_num_nd,on='mbl_num')
'''
y = train.target
x = train.drop(['mbl_num','target'],axis=1)
train_x,val_x,train_y,val_y = train_test_split(x,y,test_size=0.1,random_state=0)

dtrain = xgb.DMatrix(train_x, label=train_y)
dval = xgb.DMatrix(val_x, label=val_y)

params={
    'booster':'gbtree',
	'objective': 'binary:logistic',
	'eval_metric': 'auc',
	'max_depth':4,
	'lambda':10,
	'subsample':0.75,
	'colsample_bytree':0.75,
	'min_child_weight':2, 
	'eta': 0.025,
	'seed':0,
	'nthread':8,
        'silent':1
      }
watchlist  = [(dtrain,'train'),(dval,'val')]
#cvresult=xgb.cv(params,dtrain,num_boost_round=550,nfold=5,metrics='auc',early_stopping_rounds=100)
xgb.train(params,dtrain,num_boost_round=550,evals=watchlist)# 
'''
y = train.target
x = train.drop(['mbl_num','target'],axis=1)

test_Idx = test.mbl_num
test_x = test.drop('mbl_num',axis=1)

dtrain = xgb.DMatrix(x,label=y)
dtest = xgb.DMatrix(test_x)

params={
    'booster':'gbtree',#有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。缺省值为gbtree
	'objective': 'binary:logistic',#定义学习任务及相应的学习目标，可选的目标函数如下：reg:linear” –线性回归；“reg:logistic” –逻辑回归；“binary:logistic” –二分类的逻辑回归问题，输出为概率；
	'eval_metric': 'auc',#校验数据所需要的评价指标，不同的目标函数将会有缺省的评价指标；
	'max_depth':4,  # 树的最大深度;取值范围为：[1,∞];默认6;树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合;建议通过交叉验证（xgb.cv ) 进行调参;通常取值：3-10;
	'lambda':10,#L2 正则的惩罚系数，用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合；默认为0；
	'subsample':0.75,#用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。取值范围为：(0,1]，默认为1；
	'colsample_bytree':0.75,
	'min_child_weight':2, #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。取值范围为: [0,∞],默认1；
	'eta': 0.025,# 一个防止过拟合的参数，默认0.3;取值范围为：[0,1];通常最后设置eta为0.01~0.2;
	'seed':0,
	'nthread':8,#XGBoost运行时的线程数。缺省值是当前系统可以获得的最大线程数;如果你希望以最大速度运行，建议不设置这个参数，模型将自动获得最大线程
    'silent':1 # 打印信息的繁简指标，1表示简， 0表示繁; 建议取0，过程中的输出数据有助于理解模型以及调参。另外实际上我设置其为1也通常无法缄默运行。
      }
watchlist  = [(dtrain,'train')]
model=xgb.train(params,dtrain,num_boost_round=550,evals=watchlist)
#predict test set
test_y = model.predict(dtest)
test_result = pd.DataFrame(test_Idx,columns=["mbl_num"])
test_result["score"] = test_y
test_result.to_csv("E:/geo/code/result/xgb_result.csv",index=None,encoding='utf-8')










