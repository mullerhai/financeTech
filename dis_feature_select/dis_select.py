# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:07:54 2017

@author: luoshichao
"""
#类别不平衡  3006个正样本（违约），14611个负样本
import numpy as np
import pandas as pd
from datetime import date
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import random

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

train_num_dis=pd.read_csv('E:/geo/bank/train_data/train_num_dis.csv')
test_num_dis=pd.read_csv('E:/geo/bank/test_data/test_num_dis.csv')

train_colname=list(train_num_dis.columns)
test_num_dis=test_num_dis[train_colname]

col_names=['x_dis_'+str(item) for item in range(1,len(train_num_dis.columns))]
col_names=['mbl_num']+col_names
train_num_dis.columns=col_names
test_num_dis.columns=col_names


data=train_data_out[['mbl_num','target']]
train=pd.merge(data,train_num_dis)
train_y = train.target
train_x = train.drop(['mbl_num','target'],axis=1)
dtrain = xgb.DMatrix(train_x, label=train_y)

'''
test = pd.read_csv('../../data/test/test_x_rank.csv')
test_Idx = test.Idx
test = test.drop('Idx',axis=1)
dtest = xgb.DMatrix(test)
'''

def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
            'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'scale_pos_weight': float(len(train_y)-sum(train_y))/float(sum(train_y)),
	    'eval_metric': 'auc',
	    'gamma':gamma,
	    'max_depth':max_depth,
	    'lambda':lambd,
	    'subsample':subsample,
	    'colsample_bytree':colsample_bytree,
	    'min_child_weight':min_child_weight, 
	    'eta': 0.2,
	    'seed':random_seed,
	    'nthread':8
	 }

    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=700,evals=watchlist)
    #model.save_model('./model/xgb{0}.model'.format(iteration))
    #predict test set
    #test_y = model.predict(dtest)
    #test_result = pd.DataFrame(test_Idx,columns=["Idx"])
    #test_result["score"] = test_y
    #test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('E:/geo/code/dis_feature_select/feature_score/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)


if __name__ == "__main__":
    random_seed = list(range(10000,20000,100))
    gamma = [i/1000.0 for i in range(0,300,3)]
    max_depth = [5,6,7]
    lambd = list(range(400,600,2))
    subsample = [i/1000.0 for i in range(500,700,2)]
    colsample_bytree = [i/1000.0 for i in range(550,750,4)]
    min_child_weight = [i/1000.0 for i in range(250,550,3)]
    
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
    for i in range(5):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])