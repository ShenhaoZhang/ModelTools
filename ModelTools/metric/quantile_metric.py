import numpy as np
import pandas as pd
from sklearn import metrics

from ._base import BaseMetric

class QuantileMetric(BaseMetric):
    def __init__(
        self, 
        y_true: np.ndarray, 
        y_pred: list, 
        quantile:float,
        y_pred_name: list = None, 
        y_name: str = 'y', 
        index: pd.DatetimeIndex = None, 
        index_freq: str = None, 
        highlight: dict = None,
    ) -> None:
        self.quantile = quantile
        super().__init__(y_true, y_pred, y_pred_name, y_name, index, index_freq, highlight)
        
    def _init_data(self):
        super()._init_data(contain_resid=False)
        
    def get_metric(self,add_highlight_col=False):
        metric_dict = {}
        metric_dict['D2']      = lambda y_pred : metrics.d2_pinball_score(self.y_true,y_pred,alpha=self.quantile)
        metric_dict['PINBALL'] = lambda y_pred : metrics.mean_pinball_loss(self.y_true,y_pred,alpha=self.quantile)
        
        metric_dict = {name:list(map(func,self.y_pred)) for name,func in metric_dict.items()}
        metric = pd.DataFrame(metric_dict,index=[name.strip('Pred_') for name in self.y_pred_name]).sort_index()
        if add_highlight_col is True:
            metric = metric.assign(Highlight = lambda dt:dt.index.map(self.highlight)).fillna({'Highlight':'Others'})
        
        return metric
    
    def plot_metric_scatter(self):
        super().plot_metric_scatter(type='pca')