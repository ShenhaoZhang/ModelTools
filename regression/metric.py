from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

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
        self.init_input()
        
        self.resid = {}
        for pred_name,pred_value in self.y_pred.items():
            resid_name             = pred_name.replace('pred','resid')
            self.resid[resid_name] = self.y_true - pred_value
        
        self.data = None
        self.init_data()
    
    def init_input(self):
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
        
    def init_data(self):
        
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
        type            = 'eval',
        style_metric    = False,
        style_threshold = 0.8,
    ) -> pd.DataFrame:
        
        if type == 'eval':
            pred = self.y_pred.values()
            metric_dict = {}
            metric_dict['R2']   = lambda y_pred : metrics.r2_score(self.y_true,y_pred)
            metric_dict['MSE']  = lambda y_pred : metrics.mean_squared_error(self.y_true,y_pred)
            metric_dict['MAE']  = lambda y_pred : metrics.mean_absolute_error(self.y_true,y_pred)
            metric_dict['MBE']  = lambda y_pred : np.mean(y_pred - self.y_true)
            metric_dict['MdAE'] = lambda y_pred : metrics.median_absolute_error(self.y_true,y_pred)
            metric_dict['MAPE'] = lambda y_pred : metrics.mean_absolute_percentage_error(self.y_true,y_pred)
            metric_dict['MaxE'] = lambda y_pred : metrics.max_error(self.y_true,y_pred)
            metric_dict['SAE']  = lambda y_pred : np.std(np.abs(y_pred - self.y_true))
            metric_dict['SAPE'] = lambda y_pred : np.std(np.abs((y_pred - self.y_true) / self.y_true))
            metric_dict = {name:list(map(func,pred)) for name,func in metric_dict.items()}
            
        elif type == 'resid':
            #TODO 增加ACF特征
            resid = self.resid.values()
            metric_dict = {}
            metric_dict['Mean']   = np.mean
            metric_dict['Median'] = np.median
            metric_dict['SD']     = np.std
            metric_dict['IQR']    = lambda resid : np.quantile(resid,q=0.75) - np.quantile(resid,q=0.25)
            metric_dict['Skew']   = stats.skew
            metric_dict['Kurt']   = stats.kurtosis
            metric_dict = {name:list(map(func,resid)) for name,func in metric_dict.items()}
            
        else:
            raise TypeError('WRONG')
        
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

if __name__ == '__main__':
    y_ture = np.random.normal(size=100)
    y_pred = np.random.normal(loc=y_ture,scale=0.2)
    metric = Metric(y_ture,y_pred,y_name='abc')
    print(metric.get_metric())