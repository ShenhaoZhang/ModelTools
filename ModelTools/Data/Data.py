import itertools
from typing import Union

import numpy as np
import pandas as pd 
import plotnine as gg
import altair as alt

from .plot import ts_scatter
from ..tools.novelty import Novelty

class Data:
    
#TODO 数据清洗 探索性可视化
    
    def __init__(self,data:Union[pd.DataFrame,dict],col_x:list=None,col_y:str=None,col_ts=None) -> None:
        
        if isinstance(data,pd.DataFrame):
            self.data = data 
        if isinstance(data,dict):
            data_list = []
            for label,df in data.items():
                df['_label'] = label
                data_list.append(df)
            self.data = pd.concat(data_list,axis=0,ignore_index=True)
        self.col_x  = col_x if col_x is not None else self.data.columns
        self.col_y  = col_y
        self.col_ts = col_ts
        
    
    def get_novelty_score(self,label_train:str='train',label_test:str='test',method='lof'):
        train_x = self.data.loc[self.data._label==label_train,self.col_x]
        test_x  = self.data.loc[self.data._label==label_test,self.col_x]
        if len(train_x)==0 or len(test_x)==0:
            raise Exception(f'没有获取到{label_train}或{label_test}对应的数据')
        score = Novelty(train_x=train_x,test_x=test_x).get_score(method=method)
        return score
    
    def plot_distribution(self):
        # 直方图
        # 密度图
        # 箱行图/小提琴图
        # 山脊图
        ...
    
    def plot_ts(self):
        ...
    
    def plot_ts_scatter(self,col_x:str,col_y:str=None,label=None,add_lm=False):
        if (col_y is None) and (self.col_y is not None):
            col_y = self.col_y
        elif (col_y is None) and (self.col_y is None):
            raise Exception('缺失col_y参数')
        
        if label is not None:
            data = self.data.loc[self.data._label == label,:]
        else:
            data = self.data
        
        plot = ts_scatter(data=data,x=col_x,y=col_y,ts=self.col_ts,add_lm=add_lm)
        return plot
        
        
    def plot_scatter_matrix(self):
        pdf = []
        for a1, b1 in itertools.combinations(self.data.columns, 2):
            for (a,b) in ((a1, b1), (b1, a1)):
                sub = self.data[[a, b]].rename(columns={a: "x", b: "y"}).assign(a=a, b=b)
                pdf.append(sub)

        plot = gg.ggplot(pd.concat(pdf))
        plot += gg.geom_point(gg.aes('x','y'))
        plot += gg.facet_grid('b~a', scales='free')
        return plot
    