# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:23:12 2021

@author: DELL
"""

import numpy as np
import pandas as pd
import datetime
import calendar
import gc


backcast = pd.read_csv('E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/predicted_backcast.csv')

forecast = pd.read_csv('E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/predicted_forecast.csv')

backcast.reset_index(drop = True, inplace=True)
forecast.reset_index(drop = True, inplace=True)

backcast.drop(columns = {'Unnamed: 0'}, inplace = True)
forecast.drop(columns = {'Unnamed: 0'}, inplace = True)

final_df = pd.concat([backcast, forecast], axis=0)

final_df['Hour'] = 'h'+ final_df['usage_hour'].astype(str)
final_df['usage_date'] = pd.to_datetime(final_df['usage_date'])

final_df['year'] = final_df['usage_date'].apply(lambda x:x.year)
final_df['month'] = final_df['usage_date'].apply(lambda x:x.month)
final_df['day'] = final_df['usage_date'].apply(lambda x:x.day)

final_df.rename(columns = {'Zone': 'zone_id'}, inplace=True)

final_df2 = pd.pivot_table(final_df,values = 'Load', index=['zone_id', 'year', 'month', 'day'], columns = 'Hour')

final_df2.to_csv(r'E:/personal projects/global-energy-forecasting-competition-2012-load-forecasting/final_backcast_forecast.csv')
