# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 01:48:02 2021

@author: DELL
"""

import numpy as np
import pandas as pd
import datetime
import calendar
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import feature_assemble_functions as fe
import joblib as jb


temperature =  pd.read_csv('E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/temperature_history.csv')

temperature['Date'] = pd.to_datetime(temperature[['year','month','day']])

temperature = pd.melt(temperature, id_vars = ['station_id','Date'], value_vars = ['h'+str(i) for i in range(1,25)])

temperature['hour'] = temperature['variable'].str.split("h", expand = True)[1]

del temperature['variable']

temperature['hour'] = temperature['hour'].astype(int)

temperature['station_id'] =pd.Categorical(temperature['station_id'], ordered = False)

temperature_unmelted = temperature.pivot(index = ['Date', 'hour'], columns = 'station_id', values = 'value')

temperature_unmelted.columns = temperature_unmelted.columns.astype(str)

temperature_unmelted.reset_index(inplace = True)

temperature_unmelted.rename(columns = {'Date': 'usage_date', 'hour': 'usage_hour'}, inplace = True)

temperature_unmelted.index = temperature_unmelted[['usage_date', 'usage_hour']]

#####Checking for date range containing null values###################

temperature_unmelted[temperature_unmelted['1'].isnull()]['usage_date'].unique()

#####Concluded here that we do not have historical temperature data for 2008-06-30####

temperature_unmelted.dropna(inplace=True)

temp_new = temperature_unmelted['1']

temp_new.plot(figsize = (16,5), legend = True)



####we observe a seasonality trait in the temperature plot############

from statsmodels.tsa.stattools import adfuller

#H1: It is stationary
#H0: It is not stationary


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")



adfuller_test(temperature_unmelted['1'])

#####We conclude here that our data is stationary######################

#####We will use ARIMA to predict future temperature values############

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(temp_new.iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(temp_new.iloc[13:],lags=40,ax=ax2)


#####Creating the prediction dataframe for temperature#################

from pandas.tseries.offsets import DateOffset

future_dates=[temperature_unmelted['usage_date'][-1]+datetime.timedelta(days=1)+ DateOffset(days=x)for x in range(0,7)]

date_hours = pd.Series([i for i in range(1,25)])

future_df = {date:[i for i in range(1,25)] for date in future_dates}

future_df = pd.DataFrame(pd.concat({k: pd.Series(v) for k, v in future_df.items()}))

future_df.reset_index(drop = False, inplace=True)

future_df.drop(columns=['level_1'], inplace=True)

future_df.rename(columns={'level_0':'usage_date', 0:'usage_hour'}, inplace=True)

######fitting temperature data to ARIMA model########################################

#p=3, d=0, q=0

from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(temp_new,order=(3,0,0))

model_fit = model.fit()

model_fit.summary()

#######predict the temperatures for 2008/7/1 to 2008/7/7##############################

future_df['temperature'] = pd.Series(model_fit.forecast(steps=187)[0][19:])

new_df = pd.DataFrame()

for zone in [str(i) for i in range(1,22)]:
    future_df['Zone']=zone
    new_df = pd.concat([new_df, future_df], axis = 0)
    
####Assembling the input features#####################################################
    
fe = fe.feature_assemble_functions()

new_df = fe.day_type(new_df)    

new_df = fe.create_time_features(new_df)

for feature in fe.features:
    if feature not in new_df.columns:
        new_df[feature]=0
        
new_df['Load']=np.nan
        
for zone in new_df['Zone'].unique():
        joblib_file = str(zone)+'_model'
        model = jb.load(joblib_file)
        new_df['Load'][new_df['Zone']== zone] = model.predict(new_df[fe.features][new_df['Zone']== zone])

new_df[['Zone', 'usage_date', 'usage_hour', 'Load']].to_csv(r'E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/predicted_forecast.csv')






