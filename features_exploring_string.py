# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 12:02:29 2017

@author: luoshichao
"""

import numpy as np
import pandas as pd
from datetime import date


train_string=pd.read_csv('E:/geo/bank/train_data/train_string.csv')
test_string=pd.read_csv('E:/geo/bank/test_data/test_string.csv')
train_target=pd.read_csv('E:/geo/bank/train_data/train_data_out.csv')[['mbl_num','target']]

train_string.replace(to_replace=-999.0,value=np.nan, inplace=True)
train_string.replace(to_replace='-999.0',value=np.nan, inplace=True)

test_string.replace(to_replace=-999.0,value=np.nan, inplace=True)
test_string.replace(to_replace='-999.0',value=np.nan, inplace=True)




#第二常用城市与第二常用城市名称本质上是同一个 保留城市名称
train_string.drop(['第二常用城市','卡性质','借贷标记','第二常用城市名称'],axis=1,inplace=True)
test_string.drop(['第二常用城市','卡性质','借贷标记','第二常用城市名称'],axis=1,inplace=True)

#对卡品牌做分析
train_string['卡品牌1']=0
train_string['卡品牌1'][train_string['卡品牌']=='CUP']=1
train_string=train_string.drop(['卡品牌'],axis=1)

test_string['卡品牌1']=0
test_string['卡品牌1'][test_string['卡品牌']=='CUP']=1
test_string=test_string.drop(['卡品牌'],axis=1)


#卡名称处理
train_string['卡名称1']=0
train_string['卡名称1'][train_string['卡名称']=='金穗通宝卡(银联卡)']=1
train_string['卡名称1'][train_string['卡名称']=='龙卡通']=2
train_string['卡名称1'][train_string['卡名称']=='牡丹卡普卡']=3
train_string['卡名称1'][train_string['卡名称']=='结算通借记卡']=4
train_string['卡名称1'][train_string['卡名称']=='E时代卡']=5
train_string['卡名称1'][train_string['卡名称']=='福农灵通卡']=6
train_string['卡名称1'][train_string['卡名称']=='龙卡储蓄卡(银联卡)']=7
train_string['卡名称1'][train_string['卡名称']=='银联IC普卡']=8
train_string['卡名称1'][train_string['卡名称']=='借记IC个人普卡']=9
train_string['卡名称1'][train_string['卡名称']=='灵通卡']=10
train_string=train_string.drop(['卡名称'],axis=1)
#------------ont-hot-------------------------------
train_string=pd.get_dummies(train_string,columns=['卡名称1'])

test_string['卡名称1']=0
test_string['卡名称1'][test_string['卡名称']=='金穗通宝卡(银联卡)']=1
test_string['卡名称1'][test_string['卡名称']=='龙卡通']=2
test_string['卡名称1'][test_string['卡名称']=='牡丹卡普卡']=3
test_string['卡名称1'][test_string['卡名称']=='结算通借记卡']=4
test_string['卡名称1'][test_string['卡名称']=='E时代卡']=5
test_string['卡名称1'][test_string['卡名称']=='福农灵通卡']=6
test_string['卡名称1'][test_string['卡名称']=='龙卡储蓄卡(银联卡)']=7
test_string['卡名称1'][test_string['卡名称']=='银联IC普卡']=8
test_string['卡名称1'][test_string['卡名称']=='借记IC个人普卡']=9
test_string['卡名称1'][test_string['卡名称']=='灵通卡']=10
test_string=test_string.drop(['卡名称'],axis=1)
#------------ont-hot-------------------------------
test_string=pd.get_dummies(test_string,columns=['卡名称1'])

#对交易时间作处理

train_string=train_string.drop(['近12个月最近一笔交易时间','最近一笔交易日期','最近一笔交易时间','最近一笔交易（含转入转出）日期'],axis=1)
test_string=test_string.drop(['近12个月最近一笔交易时间','最近一笔交易日期','最近一笔交易时间','最近一笔交易（含转入转出）日期'],axis=1)

train_string['time_issame']=(train_string['近12个月最近一笔交易日期']==train_string['近12个月最近一笔交易（含转入转出）日期'])
train_string['time_issame']=train_string['time_issame'].astype(int)
train_string=train_string.drop(['近12个月最近一笔交易（含转入转出）日期'],axis=1)

test_string['time_issame']=(test_string['近12个月最近一笔交易日期']==test_string['近12个月最近一笔交易（含转入转出）日期'])
test_string['time_issame']=test_string['time_issame'].astype(int)
test_string=test_string.drop(['近12个月最近一笔交易（含转入转出）日期'],axis=1)

def to_week(x):
    x=str(x)  
    if len(x)==10: 
        return date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1
    else:
        return np.nan

def to_month(x):
    x=str(x)  
    if len(x)==10: 
        return int(x[6:8])
    else:
        return np.nan
    
def to_distance(x,y):
    x=str(x) 
    y=str(y)
    if (len(x)==10) & (len(y)==10): 
        return (date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(int(y[0:4]),int(y[4:6]),int(y[6:8]))).days
    else:
        return np.nan
        

train_string['rec12_day_of_week'] = train_string['近12个月最近一笔交易日期'].apply(to_week)
train_string['rec12_day_of_month'] = train_string['近12个月最近一笔交易日期'].apply(to_month)
train_string['his_day_of_week'] = train_string['历史最早交易日期'].apply(to_week)
train_string['his_day_of_month'] = train_string['历史最早交易日期'].apply(to_month)


train_string['time_distance'] = [to_distance(x,y) for x,y in zip(train_string['近12个月最近一笔交易日期'],train_string['历史最早交易日期'])]

test_string['rec12_day_of_week'] = test_string['近12个月最近一笔交易日期'].apply(to_week)
test_string['rec12_day_of_month'] = test_string['近12个月最近一笔交易日期'].apply(to_month)
test_string['his_day_of_week'] = test_string['历史最早交易日期'].apply(to_week)
test_string['his_day_of_month'] = test_string['历史最早交易日期'].apply(to_month)


test_string['time_distance'] = [to_distance(x,y) for x,y in zip(test_string['近12个月最近一笔交易日期'],test_string['历史最早交易日期'])]



#近12个月单笔最大交易金额对应交易类型处理
train_string['12_6_issame']=(train_string['近12个月单笔最大交易金额对应交易类型']==train_string['近6个月单笔最大交易金额对应交易类型'])
train_string['12_6_issame']=train_string['12_6_issame'].astype(int)
train_string['12_3_issame']=(train_string['近12个月单笔最大交易金额对应交易类型']==train_string['近3个月单笔最大交易金额对应交易类型'])
train_string['12_3_issame']=train_string['12_3_issame'].astype(int)
train_string['12_1_issame']=(train_string['近12个月单笔最大交易金额对应交易类型']==train_string['近1个月单笔最大交易金额对应交易类型'])
train_string['12_1_issame']=train_string['12_1_issame'].astype(int)
train_string['6_3_issame']=(train_string['近6个月单笔最大交易金额对应交易类型']==train_string['近3个月单笔最大交易金额对应交易类型'])
train_string['6_3_issame']=train_string['6_3_issame'].astype(int)
train_string['6_1_issame']=(train_string['近6个月单笔最大交易金额对应交易类型']==train_string['近1个月单笔最大交易金额对应交易类型'])
train_string['6_1_issame']=train_string['6_1_issame'].astype(int)
train_string['3_1_issame']=(train_string['近3个月单笔最大交易金额对应交易类型']==train_string['近1个月单笔最大交易金额对应交易类型'])
train_string['3_1_issame']=train_string['3_1_issame'].astype(int)

test_string['12_6_issame']=(test_string['近12个月单笔最大交易金额对应交易类型']==test_string['近6个月单笔最大交易金额对应交易类型'])
test_string['12_6_issame']=test_string['12_6_issame'].astype(int)
test_string['12_3_issame']=(test_string['近12个月单笔最大交易金额对应交易类型']==test_string['近3个月单笔最大交易金额对应交易类型'])
test_string['12_3_issame']=test_string['12_3_issame'].astype(int)
test_string['12_1_issame']=(test_string['近12个月单笔最大交易金额对应交易类型']==test_string['近1个月单笔最大交易金额对应交易类型'])
test_string['12_1_issame']=test_string['12_1_issame'].astype(int)
test_string['6_3_issame']=(test_string['近6个月单笔最大交易金额对应交易类型']==test_string['近3个月单笔最大交易金额对应交易类型'])
test_string['6_3_issame']=test_string['6_3_issame'].astype(int)
test_string['6_1_issame']=(test_string['近6个月单笔最大交易金额对应交易类型']==test_string['近1个月单笔最大交易金额对应交易类型'])
test_string['6_1_issame']=test_string['6_1_issame'].astype(int)
test_string['3_1_issame']=(test_string['近3个月单笔最大交易金额对应交易类型']==test_string['近1个月单笔最大交易金额对应交易类型'])
test_string['3_1_issame']=test_string['3_1_issame'].astype(int)

train_string['近1个月单笔最大交易金额对应交易类型1']=0
train_string['近1个月单笔最大交易金额对应交易类型1'][train_string['近1个月单笔最大交易金额对应交易类型']=='S22']=1
train_string['近1个月单笔最大交易金额对应交易类型1'][train_string['近1个月单笔最大交易金额对应交易类型']=='S24']=2

train_string['近3个月单笔最大交易金额对应交易类型1']=0
train_string['近3个月单笔最大交易金额对应交易类型1'][train_string['近3个月单笔最大交易金额对应交易类型']=='S22']=1
train_string['近3个月单笔最大交易金额对应交易类型1'][train_string['近3个月单笔最大交易金额对应交易类型']=='S24']=2


train_string['近6个月单笔最大交易金额对应交易类型1']=0
train_string['近6个月单笔最大交易金额对应交易类型1'][train_string['近6个月单笔最大交易金额对应交易类型']=='S22']=1
train_string['近6个月单笔最大交易金额对应交易类型1'][train_string['近6个月单笔最大交易金额对应交易类型']=='S24']=2



train_string['近12个月单笔最大交易金额对应交易类型1']=0
train_string['近12个月单笔最大交易金额对应交易类型1'][train_string['近12个月单笔最大交易金额对应交易类型']=='S22']=1
train_string['近12个月单笔最大交易金额对应交易类型1'][train_string['近12个月单笔最大交易金额对应交易类型']=='S24']=2

train_string=train_string.drop(['近12个月单笔最大交易金额对应交易类型','近6个月单笔最大交易金额对应交易类型',\
'近3个月单笔最大交易金额对应交易类型','近1个月单笔最大交易金额对应交易类型'],axis=1)
    
test_string['近1个月单笔最大交易金额对应交易类型1']=0
test_string['近1个月单笔最大交易金额对应交易类型1'][test_string['近1个月单笔最大交易金额对应交易类型']=='S22']=1
test_string['近1个月单笔最大交易金额对应交易类型1'][test_string['近1个月单笔最大交易金额对应交易类型']=='S24']=2

test_string['近3个月单笔最大交易金额对应交易类型1']=0
test_string['近3个月单笔最大交易金额对应交易类型1'][test_string['近3个月单笔最大交易金额对应交易类型']=='S22']=1
test_string['近3个月单笔最大交易金额对应交易类型1'][test_string['近3个月单笔最大交易金额对应交易类型']=='S24']=2


test_string['近6个月单笔最大交易金额对应交易类型1']=0
test_string['近6个月单笔最大交易金额对应交易类型1'][test_string['近6个月单笔最大交易金额对应交易类型']=='S22']=1
test_string['近6个月单笔最大交易金额对应交易类型1'][test_string['近6个月单笔最大交易金额对应交易类型']=='S24']=2



test_string['近12个月单笔最大交易金额对应交易类型1']=0
test_string['近12个月单笔最大交易金额对应交易类型1'][test_string['近12个月单笔最大交易金额对应交易类型']=='S22']=1
test_string['近12个月单笔最大交易金额对应交易类型1'][test_string['近12个月单笔最大交易金额对应交易类型']=='S24']=2

test_string=test_string.drop(['近12个月单笔最大交易金额对应交易类型','近6个月单笔最大交易金额对应交易类型',\
'近3个月单笔最大交易金额对应交易类型','近1个月单笔最大交易金额对应交易类型'],axis=1)

#近12个月交易金额排名第一的MCC

train_string['MCC_12_6_issame']=(train_string['近12个月交易金额排名第一的MCC']==train_string['近6个月交易金额排名第一的MCC'])
train_string['MCC_12_6_issame']=train_string['12_6_issame'].astype(int)


train_string['MCC_12_1_issame']=(train_string['近12个月交易金额排名第一的MCC']==train_string['近1个月交易金额排名第一的MCC'])
train_string['MCC_12_1_issame']=train_string['12_1_issame'].astype(int)

train_string['MCC_6_1_issame']=(train_string['近6个月交易金额排名第一的MCC']==train_string['近1个月交易金额排名第一的MCC'])
train_string['MCC_6_1_issame']=train_string['6_3_issame'].astype(int)

train_string['近12个月交易金额排名第一的MCC1']=0
train_string['近12个月交易金额排名第一的MCC1'][train_string['近12个月交易金额排名第一的MCC']==6011.0]=1
train_string['近12个月交易金额排名第一的MCC1'][train_string['近12个月交易金额排名第一的MCC']==5933.0]=2
train_string['近12个月交易金额排名第一的MCC1'][train_string['近12个月交易金额排名第一的MCC']==4899.0]=3


train_string['近6个月交易金额排名第一的MCC1']=0
train_string['近6个月交易金额排名第一的MCC1'][train_string['近6个月交易金额排名第一的MCC']==6011.0]=1
train_string['近6个月交易金额排名第一的MCC1'][train_string['近6个月交易金额排名第一的MCC']==5933.0]=2
train_string['近6个月交易金额排名第一的MCC1'][train_string['近6个月交易金额排名第一的MCC']==4899.0]=3


train_string['近1个月交易金额排名第一的MCC1']=0
train_string['近1个月交易金额排名第一的MCC1'][train_string['近1个月交易金额排名第一的MCC']==6011.0]=1
train_string['近1个月交易金额排名第一的MCC1'][train_string['近1个月交易金额排名第一的MCC']==5933.0]=2
train_string['近1个月交易金额排名第一的MCC1'][train_string['近1个月交易金额排名第一的MCC']==4899.0]=3

train_string=train_string.drop(['近12个月交易金额排名第一的MCC','近6个月交易金额排名第一的MCC','近1个月交易金额排名第一的MCC'],axis=1)

#近12个月交易金额排名第一的MCC

test_string['MCC_12_6_issame']=(test_string['近12个月交易金额排名第一的MCC']==test_string['近6个月交易金额排名第一的MCC'])
test_string['MCC_12_6_issame']=test_string['12_6_issame'].astype(int)


test_string['MCC_12_1_issame']=(test_string['近12个月交易金额排名第一的MCC']==test_string['近1个月交易金额排名第一的MCC'])
test_string['MCC_12_1_issame']=test_string['12_1_issame'].astype(int)

test_string['MCC_6_1_issame']=(test_string['近6个月交易金额排名第一的MCC']==test_string['近1个月交易金额排名第一的MCC'])
test_string['MCC_6_1_issame']=test_string['6_3_issame'].astype(int)

test_string['近12个月交易金额排名第一的MCC1']=0
test_string['近12个月交易金额排名第一的MCC1'][test_string['近12个月交易金额排名第一的MCC']==6011.0]=1
test_string['近12个月交易金额排名第一的MCC1'][test_string['近12个月交易金额排名第一的MCC']==5933.0]=2
test_string['近12个月交易金额排名第一的MCC1'][test_string['近12个月交易金额排名第一的MCC']==4899.0]=3


test_string['近6个月交易金额排名第一的MCC1']=0
test_string['近6个月交易金额排名第一的MCC1'][test_string['近6个月交易金额排名第一的MCC']==6011.0]=1
test_string['近6个月交易金额排名第一的MCC1'][test_string['近6个月交易金额排名第一的MCC']==5933.0]=2
test_string['近6个月交易金额排名第一的MCC1'][test_string['近6个月交易金额排名第一的MCC']==4899.0]=3


test_string['近1个月交易金额排名第一的MCC1']=0
test_string['近1个月交易金额排名第一的MCC1'][test_string['近1个月交易金额排名第一的MCC']==6011.0]=1
test_string['近1个月交易金额排名第一的MCC1'][test_string['近1个月交易金额排名第一的MCC']==5933.0]=2
test_string['近1个月交易金额排名第一的MCC1'][test_string['近1个月交易金额排名第一的MCC']==4899.0]=3

test_string=test_string.drop(['近12个月交易金额排名第一的MCC','近6个月交易金额排名第一的MCC','近1个月交易金额排名第一的MCC'],axis=1)

#对发卡行做处理

train_string['发卡行1']=0
train_string['发卡行1'][train_string['发卡行']=='建设银行']=1
train_string['发卡行1'][train_string['发卡行']=='农业银行']=2
train_string['发卡行1'][train_string['发卡行']=='工商银行']=3
train_string['发卡行1'][train_string['发卡行']=='中国银行']=4
train_string['发卡行1'][train_string['发卡行']=='招商银行']=5
train_string=train_string.drop(['发卡行'],axis=1)



test_string['发卡行1']=0
test_string['发卡行1'][test_string['发卡行']=='建设银行']=1
test_string['发卡行1'][test_string['发卡行']=='农业银行']=2
test_string['发卡行1'][test_string['发卡行']=='工商银行']=3
test_string['发卡行1'][test_string['发卡行']=='中国银行']=4
test_string['发卡行1'][test_string['发卡行']=='招商银行']=5
test_string=test_string.drop(['发卡行'],axis=1)




#近6个月最常用的交易渠道类型
train_string['近6个月最常用的交易渠道类型1']=0
train_string['近6个月最常用的交易渠道类型1'][train_string['近6个月最常用的交易渠道类型']==7.0]=1
train_string['近6个月最常用的交易渠道类型1'][train_string['近6个月最常用的交易渠道类型']==1.0]=2
train_string['近6个月最常用的交易渠道类型1'][train_string['近6个月最常用的交易渠道类型']==3.0]=3
train_string=train_string.drop(['近6个月最常用的交易渠道类型'],axis=1)


train_string['近6个月商户名是否相同']=(train_string['近6个月交易金额最大商户的商户名']==train_string['近6个月交易笔数最大商户的商户名'])
train_string['近6个月商户名是否相同']=train_string['近6个月商户名是否相同'].astype(int)

train_string['近1个月商户名是否相同']=(train_string['近1个月交易金额最大商户的商户名']==train_string['近1个月交易笔数最大商户的商户名'])
train_string['近1个月商户名是否相同']=train_string['近1个月商户名是否相同'].astype(int)

train_string=train_string.drop(['近6个月交易金额最大商户的商户名','近6个月交易笔数最大商户的商户名',\
'近1个月交易金额最大商户的商户名','近1个月交易笔数最大商户的商户名'],axis=1)
    

train_string=train_string.drop(['近12个月交易金额排名第二的MCC','近6个月交易金额排名第二的MCC','近1个月交易金额排名第二的MCC',\
 '近12个月交易金额排名第三的MCC', '近6个月交易金额排名第三的MCC','近12个月交易金额排名第四的MCC','近6个月交易金额排名第四的MCC',\
'近12个月交易金额排名第五的MCC','近6个月交易金额排名第五的MCC'],axis=1)                  


train_string=train_string.drop(['常用城市名称','近6个月交易金额排名第一的城市名称',\
                   '近3个月交易金额排名第一的城市名称','最近一笔交易城市名称',\
                   '近6个月pos交易笔数排名第一的城市名称'],axis=1)   
    
test_string['近6个月最常用的交易渠道类型1']=0
test_string['近6个月最常用的交易渠道类型1'][test_string['近6个月最常用的交易渠道类型']==7.0]=1
test_string['近6个月最常用的交易渠道类型1'][test_string['近6个月最常用的交易渠道类型']==1.0]=2
test_string['近6个月最常用的交易渠道类型1'][test_string['近6个月最常用的交易渠道类型']==3.0]=3
test_string=test_string.drop(['近6个月最常用的交易渠道类型'],axis=1)


test_string['近6个月商户名是否相同']=(test_string['近6个月交易金额最大商户的商户名']==test_string['近6个月交易笔数最大商户的商户名'])
test_string['近6个月商户名是否相同']=test_string['近6个月商户名是否相同'].astype(int)

test_string['近1个月商户名是否相同']=(test_string['近1个月交易金额最大商户的商户名']==test_string['近1个月交易笔数最大商户的商户名'])
test_string['近1个月商户名是否相同']=test_string['近1个月商户名是否相同'].astype(int)

test_string=test_string.drop(['近6个月交易金额最大商户的商户名','近6个月交易笔数最大商户的商户名',\
'近1个月交易金额最大商户的商户名','近1个月交易笔数最大商户的商户名'],axis=1)
    

test_string=test_string.drop(['近12个月交易金额排名第二的MCC','近6个月交易金额排名第二的MCC','近1个月交易金额排名第二的MCC',\
 '近12个月交易金额排名第三的MCC', '近6个月交易金额排名第三的MCC','近12个月交易金额排名第四的MCC','近6个月交易金额排名第四的MCC',\
'近12个月交易金额排名第五的MCC','近6个月交易金额排名第五的MCC'],axis=1)                  


test_string=test_string.drop(['常用城市名称','近6个月交易金额排名第一的城市名称',\
                   '近3个月交易金额排名第一的城市名称','最近一笔交易城市名称',\
                   '近6个月pos交易笔数排名第一的城市名称'],axis=1)  


train_string['省是否相同']=(train_string['当月活动省市']==train_string['常用省（市）'])
train_string['省是否相同']=train_string['省是否相同'].astype(int)
train_string=train_string.drop(['当月活动省市'],axis=1)
train_string=pd.get_dummies(train_string,columns=['常用省（市）'])

test_string['省是否相同']=(test_string['当月活动省市']==test_string['常用省（市）'])
test_string['省是否相同']=test_string['省是否相同'].astype(int)
test_string=test_string.drop(['当月活动省市'],axis=1)
test_string=pd.get_dummies(test_string,columns=['常用省（市）'])

test_colname=list(test_string.columns)
rongyu=list(set(test_colname)-set(train_string.columns))[0]
test_colname.remove(rongyu)
test_string=test_string[test_colname]


train_string['city12']=(train_string['最近一笔交易城市']==train_string['近6个月pos交易笔数排名第一的城市'])
train_string['city12']=train_string['city12'].astype(int)
train_string['city13']=(train_string['最近一笔交易城市']==train_string['近3个月交易金额排名第一的城市'])
train_string['city13']=train_string['city13'].astype(int)
train_string['city14']=(train_string['最近一笔交易城市']==train_string['近6个月交易金额排名第一的城市'])
train_string['city14']=train_string['city14'].astype(int)
train_string['city15']=(train_string['最近一笔交易城市']==train_string['常用城市'])
train_string['city15']=train_string['city15'].astype(int)


train_string['city23']=(train_string['近6个月pos交易笔数排名第一的城市']==train_string['近3个月交易金额排名第一的城市'])
train_string['city23']=train_string['city23'].astype(int)
train_string['city24']=(train_string['近6个月pos交易笔数排名第一的城市']==train_string['近6个月交易金额排名第一的城市'])
train_string['city24']=train_string['city24'].astype(int)
train_string['city25']=(train_string['近6个月pos交易笔数排名第一的城市']==train_string['常用城市'])
train_string['city25']=train_string['city25'].astype(int)

train_string['city34']=(train_string['近3个月交易金额排名第一的城市']==train_string['近6个月交易金额排名第一的城市'])
train_string['city34']=train_string['city34'].astype(int)
train_string['city35']=(train_string['近3个月交易金额排名第一的城市']==train_string['常用城市'])
train_string['city35']=train_string['city35'].astype(int)

train_string['city45']=(train_string['近6个月交易金额排名第一的城市']==train_string['常用城市'])
train_string['city45']=train_string['city45'].astype(int)

        
train_string=train_string.drop(['近6个月pos交易笔数排名第一的城市','近3个月交易金额排名第一的城市',\
                               '近6个月交易金额排名第一的城市','近6个月交易金额排名第一的城市'],axis=1) 

train_string['常用城市1']=0
train_string['常用城市1'][train_string['常用城市']==5810.0]=1
train_string['常用城市1'][train_string['常用城市']==5840.0]=2
train_string['常用城市1'][train_string['常用城市']==2900.0]=3
train_string['常用城市1'][train_string['常用城市']==6510.0]=4
train_string['常用城市1'][train_string['常用城市']==1000.0]=5
train_string['常用城市1'][train_string['常用城市']==6900.0]=6
train_string['常用城市1'][train_string['常用城市']==5210.0]=7
train_string['常用城市1'][train_string['常用城市']==6020.0]=8
train_string['常用城市1'][train_string['常用城市']==3050.0]=9
train_string['常用城市1'][train_string['常用城市']==5510.0]=10
train_string['常用城市1'][train_string['常用城市']==5880.0]=11
train_string['常用城市1'][train_string['常用城市']==7310.0]=12
train_string['常用城市1'][train_string['常用城市']==7910.0]=13
train_string['常用城市1'][train_string['常用城市']==3310.0]=14
train_string['常用城市1'][train_string['常用城市']==4910.0]=15
train_string['常用城市1'][train_string['常用城市']==6110.0]=16
train_string['常用城市1'][train_string['常用城市']==3010.0]=17
train_string['常用城市1'][train_string['常用城市']==3930.0]=18
train_string['常用城市1'][train_string['常用城市']==3320.0]=19
train_string['常用城市1'][train_string['常用城市']==3330.0]=20
train_string['常用城市1'][train_string['常用城市']==3610.0]=21
train_string['常用城市1'][train_string['常用城市']==5950.0]=22
train_string['常用城市1'][train_string['常用城市']==1100.0]=23
train_string['常用城市1'][train_string['常用城市']==7010.0]=24
train_string['常用城市1'][train_string['常用城市']==3020.0]=25
train_string['常用城市1'][train_string['常用城市']==3970.0]=26
train_string['常用城市1'][train_string['常用城市']==6030.0]=27
train_string=pd.get_dummies(train_string,columns=['常用城市1'])



test_string['city12']=(test_string['最近一笔交易城市']==test_string['近6个月pos交易笔数排名第一的城市'])
test_string['city12']=test_string['city12'].astype(int)
test_string['city13']=(test_string['最近一笔交易城市']==test_string['近3个月交易金额排名第一的城市'])
test_string['city13']=test_string['city13'].astype(int)
test_string['city14']=(test_string['最近一笔交易城市']==test_string['近6个月交易金额排名第一的城市'])
test_string['city14']=test_string['city14'].astype(int)
test_string['city15']=(test_string['最近一笔交易城市']==test_string['常用城市'])
test_string['city15']=test_string['city15'].astype(int)


test_string['city23']=(test_string['近6个月pos交易笔数排名第一的城市']==test_string['近3个月交易金额排名第一的城市'])
test_string['city23']=test_string['city23'].astype(int)
test_string['city24']=(test_string['近6个月pos交易笔数排名第一的城市']==test_string['近6个月交易金额排名第一的城市'])
test_string['city24']=test_string['city24'].astype(int)
test_string['city25']=(test_string['近6个月pos交易笔数排名第一的城市']==test_string['常用城市'])
test_string['city25']=test_string['city25'].astype(int)

test_string['city34']=(test_string['近3个月交易金额排名第一的城市']==test_string['近6个月交易金额排名第一的城市'])
test_string['city34']=test_string['city34'].astype(int)
test_string['city35']=(test_string['近3个月交易金额排名第一的城市']==test_string['常用城市'])
test_string['city35']=test_string['city35'].astype(int)

test_string['city45']=(test_string['近6个月交易金额排名第一的城市']==test_string['常用城市'])
test_string['city45']=test_string['city45'].astype(int)

        
test_string=test_string.drop(['近6个月pos交易笔数排名第一的城市','近3个月交易金额排名第一的城市',\
                               '近6个月交易金额排名第一的城市','近6个月交易金额排名第一的城市'],axis=1) 

test_string['常用城市1']=0
test_string['常用城市1'][test_string['常用城市']==5810.0]=1
test_string['常用城市1'][test_string['常用城市']==5840.0]=2
test_string['常用城市1'][test_string['常用城市']==2900.0]=3
test_string['常用城市1'][test_string['常用城市']==6510.0]=4
test_string['常用城市1'][test_string['常用城市']==1000.0]=5
test_string['常用城市1'][test_string['常用城市']==6900.0]=6
test_string['常用城市1'][test_string['常用城市']==5210.0]=7
test_string['常用城市1'][test_string['常用城市']==6020.0]=8
test_string['常用城市1'][test_string['常用城市']==3050.0]=9
test_string['常用城市1'][test_string['常用城市']==5510.0]=10
test_string['常用城市1'][test_string['常用城市']==5880.0]=11
test_string['常用城市1'][test_string['常用城市']==7310.0]=12
test_string['常用城市1'][test_string['常用城市']==7910.0]=13
test_string['常用城市1'][test_string['常用城市']==3310.0]=14
test_string['常用城市1'][test_string['常用城市']==4910.0]=15
test_string['常用城市1'][test_string['常用城市']==6110.0]=16
test_string['常用城市1'][test_string['常用城市']==3010.0]=17
test_string['常用城市1'][test_string['常用城市']==3930.0]=18
test_string['常用城市1'][test_string['常用城市']==3320.0]=19
test_string['常用城市1'][test_string['常用城市']==3330.0]=20
test_string['常用城市1'][test_string['常用城市']==3610.0]=21
test_string['常用城市1'][test_string['常用城市']==5950.0]=22
test_string['常用城市1'][test_string['常用城市']==1100.0]=23
test_string['常用城市1'][test_string['常用城市']==7010.0]=24
test_string['常用城市1'][test_string['常用城市']==3020.0]=25
test_string['常用城市1'][test_string['常用城市']==3970.0]=26
test_string['常用城市1'][test_string['常用城市']==6030.0]=27
test_string=pd.get_dummies(test_string,columns=['常用城市1'])

train_string.replace(to_replace=np.nan,value=-999.0, inplace=True)
train_string.to_csv('E:/geo/bank/train_data/train_string_nan.csv',index=None,encoding='utf-8') 


test_string.replace(to_replace=np.nan,value=-999.0, inplace=True)
test_string.to_csv('E:/geo/bank/test_data/test_string_nan.csv',index=None,encoding='utf-8') 