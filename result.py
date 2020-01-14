# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 08:33:26 2017

@author: luoshichao
"""

import pandas as pd
xgb_result=pd.read_csv('E:/geo/code/result/xgb_result.csv')
test_all_data=pd.read_csv('E:/geo/test_all_data.csv',low_memory=False)[['name','mbl_num','card']]
result_end=pd.merge(test_all_data,xgb_result,on='mbl_num')
result_end.to_csv('E:/geo/code/result/result_end.csv',index=None,encoding='utf-8')