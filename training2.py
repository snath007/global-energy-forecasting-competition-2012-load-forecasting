# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 02:31:19 2021

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


data = pd.read_csv('E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/train_pred_data.csv')

data.columns

data.drop(columns = ['Unnamed: 0'], inplace = True)

data['usage_date'] = pd.to_datetime(data['usage_date'])

data['Zone'] = data['Zone'].astype(str)

data.dropna(inplace = True)

data['Load'] = data['Load'].astype(float)

#####train test split#######

train = data[data['usage_date']<'2008-01-01']
test = data[data['usage_date']>='2008-01-01']

#####training models for each zone#############################


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_models(zone, df):
    regr = GradientBoostingRegressor(max_depth=2, random_state=0, n_estimators = 300)
    X =  df[features][df['Zone']== zone]
    y = df[df['Zone']== zone]['Load']
    model = regr.fit(X,y)
    print('training model for zone' + str(zone))
    return model
    

model_dict = {}

for zone in train['Zone'].unique():
    model_dict[zone]=train_models(zone, train)
    
#######create models for prediction (NOT FOR MODEL PERFORMANCE)#######

def train_and_save_models(zone, df):
    regr = GradientBoostingRegressor(max_depth=2, random_state=0, n_estimators = 300)
    X =  df[features][df['Zone']== zone]
    y = df[df['Zone']== zone]['Load']
    model = regr.fit(X,y)
    joblib_file = str(zone)+'_model'
    print('saving model for zone' + str(zone))
    jb.dump(model, joblib_file)
    


for zone in data['Zone'].unique():
    train_and_save_models(zone, data)  
    
    
#######create test results for model performance (MAPE)#########

def create_test_results(df):
    for zone in df['Zone'].unique():
        df['pred'][df['Zone']== zone] = model_dict[zone].predict(df[features][df['Zone']== zone])
        
    return df        


test['pred'] = np.nan
test = create_test_results(test)

####Calculating MAPE of the model################################

test['abs'] = abs(test['pred']-test['Load'])

def plot_zone_graphs(df, zone):
    temp_df = df[df['Zone']==zone].tail(168)
    temp_df.index = temp_df[['usage_date', 'usage_hour']]
    temp_df[['Load','pred']].plot(figsize=(12,8))
    
plot_zone_graphs(test, '21')



test_grouped = test.groupby(['Zone', 'usage_date']).agg({'Load': 'mean', 'pred': 'mean', 'abs': 'mean'})
test_grouped['mape'] = (test_grouped['abs']/test_grouped['Load'])*100 


MAPE = pd.pivot_table(test_grouped, values= 'mape', index = 'usage_date', columns = 'Zone', aggfunc='sum')

print(MAPE)

MAPE.to_csv(r'E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/global_forecast_mape.csv')

gc.collect()





    












