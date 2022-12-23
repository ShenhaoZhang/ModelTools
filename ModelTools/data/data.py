import itertools
from typing import Union

import numpy as np
import pandas as pd 

from ..plot.matrix_scatter import matrix_scatter
from ..plot.ts2_scatter import ts2_scatter

class Data:
    
    def __init__(self,data:Union[pd.DataFrame,dict],col_x:list=None,col_y:str=None,col_ts=None) -> None:
        self.data = data 
        self._init_data()
        self.col_x  = col_x if col_x is not None else self.data.columns
        self.col_y  = col_y
        self.col_ts = col_ts
    
    def _init_data(self):
        if isinstance(self.data,dict):
            data_list = []
            for label,df in self.data.items():
                df['_label'] = label
                data_list.append(df)
            self.data = pd.concat(data_list,axis=0,ignore_index=True)
            self.is_labeled = True
    
    def plot_distribution(self):
        # 直方图
        # 密度图
        # 箱行图/小提琴图
        # 山脊图
        ...
    
    def plot_ts(self):
        ...
    
    def plot_ts2_scatter(self,col_x:str,col_y:str,col_ts=None):
        plot = ts2_scatter(data=self.data,x=col_x,y=col_y,ts=col_ts)
        return plot
        
        
    def plot_scatter_matrix(self):
        plot = matrix_scatter(self.data)
        return plot
    