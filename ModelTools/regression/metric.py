from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from .plot.metric_plotnine import plot_TVP,plot_resid

class Metric:
    """
    预测效果的评估指标
    """
    
    def __init__(
        self, 
        y_true     : np.ndarray,
        y_pred     : Union[np.ndarray,list,dict],
        y_name     : str              = 'y',
        index      : pd.DatetimeIndex = None,
        index_freq : str              = None,
    ) -> None:
        
        self.y_true = np.array(y_true)
        
        # 初始化预测值
        if isinstance(y_pred,np.ndarray):
            self.y_pred = {f'pred_{y_name}':y_pred}
        elif isinstance(y_pred,list):
            pred_name = [f'pred_{i}' for i in range(len(y_pred))]
            self.y_pred = dict(zip(pred_name,y_pred))
        elif isinstance(y_pred,dict):
            self.y_pred = {f'pred_{name}':value for name,value in y_pred.items()}
        else:
            raise Exception('WRONG')
        
        self.sample_n   = len(self.y_true)
        self.y_pred_n   = len(self.y_pred)
        self.y_name     = y_name
        self.index      = index
        self.index_freq = index_freq
        self._init_input()
        
        self.resid = {}
        for pred_name,pred_value in self.y_pred.items():
            resid_name             = pred_name.replace('pred','resid')
            self.resid[resid_name] = self.y_true - pred_value
        
        self.data = None
        self._init_data()
    
    def _init_input(self):
        # 真实值与预测值的的长度校验
        for pred in self.y_pred.values():
            if len(pred) != len(self.y_true):
                raise ValueError('WRONG')
            if np.isnan(pred).any():
                raise ValueError('WRONG')

        # 当索引不为None时的校验
        if self.index is not None:
            self.index_name = 'datetime'
            if not isinstance(self.index,pd.DatetimeIndex):
                try:
                    self.index = pd.DatetimeIndex(self.index)
                except Exception as E:
                    print(E)
                    raise TypeError('WRONG')
            if self.index_freq is None:
                raise ValueError('WRONG')
            if len(self.index) != len(self.y_true):
                raise ValueError('WRONG')
        else:
            self.index_name = 'sample_order'
            self.index      = np.arange(self.sample_n)
        
    def _init_data(self):
        
        true_df  = pd.DataFrame(data = {self.y_name:self.y_true},index=self.index)
        pred_df  = pd.DataFrame(self.y_pred)
        resid_df = pd.DataFrame(self.resid)
        
        self.data = (
            pd.concat([true_df,pred_df,resid_df],axis=1)
            .reset_index()
            .pipe(
                pd.wide_to_long,
                stubnames = ['pred','resid'],
                i         = ['index',self.y_name],
                j         = 'Method',
                suffix    = '\w+',
                sep       = '_'
            )
            .reset_index()
            .infer_objects()
            .set_index('index')
        )
        
        # 时间索引填充空缺值
        if isinstance(self.index,pd.DatetimeIndex):
            time_start     = self.data.Time.min()
            time_end       = self.data.Time.max()
            complete_index = pd.MultiIndex.from_product(
                [
                    pd.date_range(time_start,time_end,freq=self.index_freq), 
                    self.data.Method.drop_duplicates()
                ],
                names=['Time','Method']
            )
            self.data = (
                self.data
                .set_index(['Time','Method'])
                .reindex(complete_index)
                .reset_index()
            )
    
    def get_metric(
        self,
        metric          = ['R2','MAE','MAPE','Mean','Std','Skew','Kurt','Median','IQR'],
        style_metric    = False,
        style_threshold = 0.8,
    ) -> pd.DataFrame:
        
        metric_dict  = {}
        y_pred :list = self.y_pred.values()
        y_resid:list = self.resid.values()
        
        for metric_name in metric:
            if metric_name == 'R2':
                metric_dict[metric_name] = [metrics.r2_score(self.y_true,pred) for pred in y_pred]
            elif metric_name == 'MSE':
                metric_dict[metric_name] = [metrics.mean_squared_error(self.y_true,pred) for pred in y_pred]
            elif metric_name == 'MAE':
                metric_dict[metric_name] = [metrics.mean_absolute_error(self.y_true,pred) for pred in y_pred]
            elif metric_name == 'MAPE':
                metric_dict[metric_name] = [metrics.mean_absolute_percentage_error(self.y_true,pred) for pred in y_pred]
            elif metric_name == 'Mean':
                metric_dict[metric_name] = [np.mean(resid) for resid in y_resid]
            elif metric_name == 'Std':
                metric_dict[metric_name] = [np.std(resid) for resid in y_resid]
            elif metric_name == 'Skew':
                metric_dict[metric_name] = [stats.skew(resid) for resid in y_resid]
            elif metric_name == 'Kurt':
                metric_dict[metric_name] = [stats.kurtosis(resid) for resid in y_resid]
            elif metric_name == 'Median':
                metric_dict[metric_name] = [np.median(resid) for resid in y_resid]
            elif metric_name == 'IQR':
                metric_dict[metric_name] = [stats.iqr(resid) for resid in y_resid]
            else:
                raise Exception(f'{metric_name} not support')
        
        metric = pd.DataFrame(
            data  = metric_dict,
            index = [name.strip('pred_') for name in self.y_pred.keys()]
            ).sort_index()
        
        if style_metric:
            metric = self.style_metric(metric=metric,threshold=style_threshold)
            
        return metric
    
    @staticmethod    
    def style_metric(metric:pd.DataFrame,threshold=0.8,up=None,dw=None):
        def style(sr:pd.Series):
            if sr.name in ['R2','D2']:
                sr = sr
            elif sr.name in ['MBE']:
                sr = -abs(sr)
            else:
                sr = -sr
            
            if up is None and dw is None:
                sr_up = sr.quantile(q=threshold)
                sr_dw = sr.quantile(q=1-threshold)
            else:
                sr_up=up
                sr_dw=dw
            
            style = np.where(sr>sr_up,'color: red;','opacity: 20%;')
            style = np.where(sr<sr_dw,'color:green',style)
            return style
        metric = metric.style.apply(lambda sr:style(sr))
        return metric
    
    def plot_TVP(self, error = 0.1, error_type = 'pct', type='scatter'):
            
        plot = plot_TVP(
            data       = self.data,
            y_name     = self.y_name,
            error      = error,
            error_type = error_type,
            type       = type
        )
            
        return plot
    
    def plot_resid(self,type='index',var:pd.DataFrame=None):
        plot = plot_resid(
            data = self.data,
            type = type,
            var  = var
        )
        
        return plot



if __name__ == '__main__':
    y_ture = np.random.normal(size=100)
    y_pred = np.random.normal(loc=y_ture,scale=0.2)
    metric = Metric(y_ture,[y_pred,y_pred*2],y_name='y_input')
    print(metric.data.reset_index())
    print(metric.get_metric())