import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from ._base import BaseMetric

class QuantileMetric(BaseMetric):
    def __init__(
        self, 
        y_true: np.ndarray, 
        y_pred: list, 
        y_pred_name: list = None, 
        y_name: str = 'y', 
        index: pd.DatetimeIndex = None, 
        index_freq: str = None, 
        highlight: dict = None
    ) -> None:
        super().__init__(y_true, y_pred, y_pred_name, y_name, index, index_freq, highlight)
        
    def _init_outlier(self):
        #TODO 更改
        # 残差及残差中的异常值
        self.resid           = [self.y_true-pred for pred in self.y_pred]
        resid_total          = np.array(self.resid).flatten()
        self.resid_total_iqr = np.quantile(resid_total,q=0.75) - np.quantile(resid_total,q=0.25)
        
        # 异常值的定义是残差大于1.5倍IQR的预测样本点
        self.outlier_index = [np.abs(self.resid[i]) > (1.5 * self.resid_total_iqr) for i in range(self.y_pred_n)]
        self.outlier_count = [np.sum(self.outlier_index[i]) for i in range(self.y_pred_n)]
        self.outlier_pct   = [np.round(self.outlier_count[i] / self.sample_n,4) for i in range(self.y_pred_n)]
        
    def get_metric(self,type='eval',add_highlight_col=False):
        #TODO 更改
        if type == 'eval':
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
            
            metric_dict = {name:list(map(func,self.y_pred)) for name,func in metric_dict.items()}
            
        elif type == 'resid':
            metric_dict = {}
            metric_dict['Mean']   = np.mean
            metric_dict['Median'] = np.median
            metric_dict['SD']     = np.std
            metric_dict['IQR']    = lambda resid : np.quantile(resid,q=0.75) - np.quantile(resid,q=0.25)
            metric_dict['Skew']   = stats.skew
            metric_dict['Kurt']   = stats.kurtosis
            
            metric_dict = {'resid_'+name:list(map(func,self.resid)) for name,func in metric_dict.items()}
            metric_dict['resid_Outlier_pct'] = self.outlier_pct
        else:
            raise TypeError('WRONG')
        
        metric = pd.DataFrame(metric_dict,index=[name.strip('Pred_') for name in self.y_pred_name]).sort_index()
        if add_highlight_col is True:
            metric = metric.assign(Highlight = lambda dt:dt.index.map(self.highlight)).fillna({'Highlight':'Others'})
        
        return metric