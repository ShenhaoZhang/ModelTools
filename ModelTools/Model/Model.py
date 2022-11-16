# 可自定义PIPELINE
# 可依次添加PIPELINE
import numpy as np
import pandas as pd
import plotnine as gg
from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV

from base_model import get_model

class Model:
    def __init__(self,data:pd.DataFrame,col_x:list,col_y:str) -> None:
        self.data = data 
        self.col_x = col_x
        self.col_y = col_y
        
        self.train_data, self.test_data = train_test_split(data, test_size=0.3, random_state=0, shuffle=False)
        self._fit_model()
    
    def _fit_model(self):
        ...
    
    def add_model(self):
        ...
    
    def choose_model(self):
        ...
    
