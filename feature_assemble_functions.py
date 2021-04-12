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
    
    
