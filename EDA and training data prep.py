# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:17:37 2021

@author: DELL
"""

import numpy as np
import pandas as pd
import datetime
import calendar
import gc
import matplotlib.pyplot as plt
import seaborn as sns

######import feature assemble class#########

import feature_assemble_functions as m

##load history data EDA###

settlement = pd.read_csv('E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/Load_history.csv')

settlement.columns

settlement.info()

settlement.isnull().sum()

settlement['Date'] = pd.to_datetime(settlement[['year','month','day']])


settlement_long = pd.melt(settlement, id_vars = ['zone_id','Date'], value_vars = ['h'+str(i) for i in range(1,25)])

settlement_long['Date'][settlement_long['value'].isnull()].unique()

####Concluded here that the missing settlement loads are for the 8 weeks which we have to create the backcast and 1 week which we need to forecast########

settlement_long['hour'] = settlement_long['variable'].str.split("h", expand = True)[1]

del settlement_long['variable']

settlement_long.rename(columns = {'zone_id': 'Zone', 'value': 'Load'}, inplace = True)

settlement_long.rename(columns = {'Date': 'usage_date', 'hour': 'usage_hour'}, inplace = True)


fn = m.feature_assemble_functions()

settlement_long = fn.day_type(settlement_long)

settlement_long['Zone'] = settlement_long['Zone'].astype(str)

#settlement_long.sort_values(by = ['hour', 'Zone', 'Date'], ascending = [True, True, True], inplace = True)

# settlement_long2 = settlement_long.dropna()

settlement_long['Load'] = settlement_long['Load'].astype(str)

settlement_long['Load'] = settlement_long['Load'].str.replace(',', '')

settlement_long['Load'] = settlement_long['Load'].astype(float)


# settlement_long['Load'] = settlement_long['Load'].astype(int)


settlement_long['usage_hour'] = settlement_long['usage_hour'].astype(int)

settlement_long2 = settlement_long.groupby(['usage_date', 'usage_hour']).agg({'Load':'sum'})

settlement_long2.reset_index(inplace = True, drop=False)

settlement_long2 = fn.day_type(settlement_long2)

settlement_long2['Zone'] = str(21)

settlement_long3 = pd.concat([settlement_long, settlement_long2], axis=0)


settlement_long3 = fn.time_cycle(settlement_long3)

# settlement_long2.columns

# settlement_long3 = settlement_long2[['Zone', 'sin_year','cos_year', 'sin_hour', 'cos_hour', 'Load', 'day_type']]

# settlement_long3['Zone'] = pd.Categorical(settlement_long3['Zone'], ordered = False)

#####Historical temperature data EDA##############

temperature =  pd.read_csv('E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/temperature_history.csv')

temperature.columns

temperature['Date'] = pd.to_datetime(temperature[['year','month','day']])

temperature = pd.melt(temperature, id_vars = ['station_id','Date'], value_vars = ['h'+str(i) for i in range(1,25)])

temperature.isnull().sum()

temperature['Date'][temperature['value'].isnull()].unique()
temperature['variable'][temperature['value'].isnull()].unique()

##Here we observe that for 2020-06-30 we have actual temperature till 6th hour##

temperature.info()

temperature['hour'] = temperature['variable'].str.split("h", expand = True)[1]

del temperature['variable']

temperature['hour'] = temperature['hour'].astype(int)

temperature['station_id'] =pd.Categorical(temperature['station_id'], ordered = False)

temperature_unmelted = temperature.pivot(index = ['Date', 'hour'], columns = 'station_id', values = 'value')

temperature_unmelted.columns = temperature_unmelted.columns.astype(str)

temperature_unmelted.reset_index(inplace = True)

len(settlement_long2['usage_date'].unique())
len(temperature_unmelted['Date'].unique())

date_comp = [i for i in temperature_unmelted['Date'].unique() if i not in settlement_long2['usage_date'].unique() ]

####We display those set of dates for which we have historical temperature but not settlement####

for i in date_comp:
    print(i)

##############

temperature_unmelted.rename(columns = {'Date': 'usage_date', 'hour': 'usage_hour'}, inplace = True)

merged_data = pd.merge(settlement_long3, temperature_unmelted, on = ['usage_date', 'usage_hour'], how = 'left')

merged_data['Zone'] = pd.Categorical(merged_data['Zone'], ordered = False)

merged_data.info()

merged_data.isnull().sum()


plt.figure(figsize=(14,12))
sns.heatmap(merged_data[['Zone','Load','day_type','sin_year', 'cos_year', 'sin_hour', 'cos_hour', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']].corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);

####We observe that the historical temperature of various weather stations are highly correlated to one another, so we keep temperature data for one station only####

merged_data.drop(columns=['2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], inplace = True)

merged_data['usage_date'] = pd.to_datetime(merged_data['usage_date'])

merged_data = fn.create_time_features(merged_data)

plt.figure(figsize=(14,12))
sns.heatmap(merged_data[['Zone','Load','day_type','sin_year', 'cos_year', 'sin_hour', 'cos_hour', '1']].corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);

merged_data.rename(columns = {'1': 'temperature'}, inplace = True)

merged_data.to_csv(r'E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/train_pred_data.csv')


gc.collect()



