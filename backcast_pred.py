# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:05:01 2021

@author: DELL
"""

import numpy as np
import pandas as pd
import datetime
import calendar
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb

import feature_assemble_functions as fe


df = pd.read_csv('E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/train_pred_data.csv')


##########Assembling prediction/backcast dataframe##################


#Following dates are to be backcasted:

# 2005/3/6 - 2005/3/12;

# 2005/6/20 - 2005/6/26;

# 2005/9/10 - 2005/9/16;

# 2005/12/25 - 2005/12/31;

# 2006/2/13 - 2006/2/19;

# 2006/5/25 - 2006/5/31;

# 2006/8/2 - 2006/8/8;

# 2006/11/22 - 2006/11/28;


df = df[((df['usage_date']>='2005-03-06')&(df['usage_date']<='2005-03-12')) | ((df['usage_date']>='2005-06-20')&(df['usage_date']<='2005-06-26')) | ((df['usage_date']>='2005-12-25')&(df['usage_date']<='2005-12-31')) | ((df['usage_date']>='2006-02-13')&(df['usage_date']<='2006-02-19')) | ((df['usage_date']>='2006-05-25')&(df['usage_date']<='2006-05-31')) | ((df['usage_date']>='2006-08-02')&(df['usage_date']<='2006-08-08')) | ((df['usage_date']>='2006-11-22')&(df['usage_date']<='2006-11-28')) | ((df['usage_date']>='2005-09-10')&(df['usage_date']<='2005-09-16'))]


#####Generate predictions######

fe = fe.feature_assemble_functions()

features = fe.features

for zone in df['Zone'].unique():
        joblib_file = str(zone)+'_model'
        model = jb.load(joblib_file)
        df['Load'][df['Zone']== zone] = model.predict(df[features][df['Zone']== zone])
        

df[['Zone', 'usage_date', 'usage_hour', 'Load']].to_csv(r'E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/predicted_backcast.csv')

















