from typing import Union

import numpy as np
import pandas as pd
import lets_plot as gg 
gg.LetsPlot.setup_html()

from .linear_model import LinearModel
from .metric import Metric

# 排序方法：随机 时间 某个工况顺序 基于某个样本全部工况相似程度

class ModelVal:
    def __init__(
        self,
        models     : list,
        data       : pd.DataFrame,
        random_seed: int = 1
    ) -> None:
        
        self.data        = data
        self.models      = self._init_models(models)
        self.random_seed = random_seed
    
    def _init_models(self,models) -> dict:
        model_res = {} # mod_name:mod
        
        if isinstance(models,list):
            for mod in models:
                if isinstance(mod,str):
                    model_res[mod] = mod
                    self.col_y = mod.split('~')[0]
        
        return model_res
    
    def _predict_test_data(self,train_data,test_data,**kwargs) -> dict:
        predict = {}
        for mod_name,mod in self.models.items():
            if isinstance(mod,str):
                mod               = LinearModel(formula=mod,data=train_data,show_progress=False).fit(**kwargs,n_bootstrap=0)
                predict[mod_name] = mod._predict(new_data=test_data)
        return predict
    
    def _get_metric_by_split(self,split_idx):
        all_metric = {}
        for idx in split_idx:
            train_data    = self.data.iloc[:idx,:]
            test_data     = self.data.iloc[idx:,:]
            predict       = self._predict_test_data(train_data,test_data)
        
            idx_metric = Metric(
                y_true=test_data.loc[:,self.col_y].values,
                y_pred=predict
            )
            
            all_metric[idx] = idx_metric.get_metric()
        
        all_metric = (
            pd.concat(all_metric,axis=0)
            .reset_index()
            .rename(columns={'level_0':'idx','level_1':'mod'})
        )
        return all_metric
    
    def val_by_random(self):
        ...
    
    def val_by_index(self,start:Union[int,float],step:int,min_test_sample=30):
        start_idx         = start if start >= 1 else int(len(self.data)*start)
        split_idx         = range(start_idx,len(self.data)-min_test_sample,step)
        self.metric_index = self._get_metric_by_split(split_idx=split_idx)
        return self

    def val_by_x(self,x_name,start:Union[int,float],step:int,aesc=True,min_test_sample=30):
        ...
    
    def val_by_sim_x(self):
        ...

    def plot_metric(self,type='index',metric=['MAE','MAPE'],log_metric=False):
        if type == 'index':
            metric_data = self.metric_index
        
        plot = (
            metric_data
            .melt(id_vars=['idx','mod'],var_name='metric',value_name='value')
            .loc[lambda dt:dt.metric.isin(metric)]
            .pipe(gg.ggplot)
            + gg.aes(x='idx',y='value',color='mod')
            + gg.geom_line()
            + gg.facet_wrap(facets='metric',scales='free_y')
            + gg.ggsize(1000,600)
        )
        
        if log_metric:
            plot += gg.scale_y_log10()
        
        return plot