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
    
  
    
    
    def day_type(self, df):
        
        df['day_type'] = 1
        df['day_type'] = np.where((df["usage_date"].dt.dayofweek == 5) | (df["usage_date"].dt.dayofweek == 6), 0, df["day_type"])
        
        return df
    
    features = ['day_type', 'sin_year', 'cos_year', 'sin_hour', 'cos_hour', 'temperature', 'woy_1', 'woy_10', 'woy_11', 'woy_12', 'woy_13', 'woy_14', 'woy_15', 'woy_16', 'woy_17', 'woy_18', 'woy_19', 'woy_2', 'woy_20', 'woy_21', 'woy_22', 'woy_23', 'woy_24', 'woy_25', 'woy_26', 'woy_27', 'woy_28', 'woy_29', 'woy_3', 'woy_30', 'woy_31', 'woy_32', 'woy_33', 'woy_34', 'woy_35', 'woy_36', 'woy_37', 'woy_38', 'woy_39', 'woy_4', 'woy_40', 'woy_41', 'woy_42', 'woy_43', 'woy_44', 'woy_45', 'woy_46', 'woy_47', 'woy_48', 'woy_49', 'woy_5', 'woy_50', 'woy_51', 'woy_52', 'woy_53', 'woy_6', 'woy_7', 'woy_8', 'woy_9', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6', 'dow_7']

