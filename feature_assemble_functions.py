# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:51:08 2021

@author: DELL
"""

import numpy as np
import pandas as pd
import calendar

class feature_assemble_functions():
    
        
    def __init__(self):
        pass
    
    
    def time_cycle(self,df):
        
        def yearly_cycle(usage_date, trig_func):
            usage_date = pd.to_datetime(usage_date, format='%Y-%m-%d')
            day_number = usage_date.dayofyear # cycles from 1 to 365/6
            if calendar.isleap(usage_date.year):
                days_year = 366.0
            else:
                days_year = 365.0
    
            year_cycle = trig_func(2*np.pi * (day_number) / days_year)
    
            return year_cycle
        
        
        def daily_cycle(usage_hour, trig_func):
        
            day_cycle = trig_func(2*np.pi * (usage_hour) / 24.0)
    
            return day_cycle
        
        df = df.reset_index()
        df['sin_year'] = df.usage_date.apply(lambda x: yearly_cycle(x, np.sin))
        df['cos_year'] = df.usage_date.apply(lambda x: yearly_cycle(x, np.cos))
    
        df['sin_hour'] = df.usage_hour.apply(lambda x: daily_cycle(x, np.sin))
        df['cos_hour'] = df.usage_hour.apply(lambda x: daily_cycle(x, np.cos))
        
    #    features = features + ['sin_year', 'cos_year', 'sin_hour', 'cos_hour']
        
        return df
    
    
    def create_time_features(self, df):
        
        df['woy'] = df['usage_date'].apply(lambda x:'woy_'+ str(x.isocalendar()[1]))
        df['dow'] = df['usage_date'].apply(lambda x:'dow_'+ str(x.isocalendar()[2]))
        
        df['woy'] = pd.Categorical(df['woy'], ordered=False)
        df['dow'] = pd.Categorical(df['dow'], ordered=False)
        
        woy = pd.get_dummies(df['woy'])
        dow = pd.get_dummies(df['dow'])
        
        df = pd.concat([df, woy], axis=1)
        df = pd.concat([df, dow], axis=1)
        
        df.drop(columns=['woy','dow'], inplace=True)
        
        return df
    
    
    
    def day_type(self, df):
        
        df['day_type'] = 1
        df['day_type'] = np.where((df["usage_date"].dt.dayofweek == 5) | (df["usage_date"].dt.dayofweek == 6), 0, df["day_type"])
        
        return df
    
    features = ['day_type', 'sin_year', 'cos_year', 'sin_hour', 'cos_hour', 'temperature', 'woy_1', 'woy_10', 'woy_11', 'woy_12', 'woy_13', 'woy_14', 'woy_15', 'woy_16', 'woy_17', 'woy_18', 'woy_19', 'woy_2', 'woy_20', 'woy_21', 'woy_22', 'woy_23', 'woy_24', 'woy_25', 'woy_26', 'woy_27', 'woy_28', 'woy_29', 'woy_3', 'woy_30', 'woy_31', 'woy_32', 'woy_33', 'woy_34', 'woy_35', 'woy_36', 'woy_37', 'woy_38', 'woy_39', 'woy_4', 'woy_40', 'woy_41', 'woy_42', 'woy_43', 'woy_44', 'woy_45', 'woy_46', 'woy_47', 'woy_48', 'woy_49', 'woy_5', 'woy_50', 'woy_51', 'woy_52', 'woy_53', 'woy_6', 'woy_7', 'woy_8', 'woy_9', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6', 'dow_7']

