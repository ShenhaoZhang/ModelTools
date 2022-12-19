import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from ..plot import plot_gg
from ..plot import plot_alt
from ._base import BaseMetric
from ..plot.ts_line import ts_line

class CentralMetric(BaseMetric):
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
        self.resid           = [self.y_true-pred for pred in self.y_pred]
        self.main_metric = 'MSE'
        self.plot_show_metric = list(self.highlight.keys())
        if len(self.plot_show_metric) < 6:
            add_metric_count = 6 - len(self.plot_show_metric)
            add_metric = (
                self.metric()
                .loc[lambda dt:~dt.index.isin(self.plot_show_metric)]
                .sort_values(by=self.main_metric,ascending=True)
                .head(add_metric_count)
                .index.to_list()
            )
            self.plot_show_metric += add_metric
        
    def _init_data(self):
        super()._init_data(contain_resid=True)
    
    def metric(self,type='eval') -> pd.DataFrame:
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
            #TODO 增加ACF特征
            metric_dict = {}
            metric_dict['Mean']   = np.mean
            metric_dict['Median'] = np.median
            metric_dict['SD']     = np.std
            metric_dict['IQR']    = lambda resid : np.quantile(resid,q=0.75) - np.quantile(resid,q=0.25)
            metric_dict['Skew']   = stats.skew
            metric_dict['Kurt']   = stats.kurtosis
            metric_dict = {'resid_'+name:list(map(func,self.resid)) for name,func in metric_dict.items()}
        else:
            raise TypeError('WRONG')
        metric = pd.DataFrame(metric_dict,index=[name.strip('Pred_') for name in self.y_pred_name]).sort_index()
        return metric
    
    #TODO 可选日期作为y轴
    def plot_TvP(
        self,
        show_metric:list = None, 
        add_lm=False, 
        add_outlier=False, 
        add_quantile=False, 
        figure_size=(10, 5), 
        scales='fixed', 
    ):
        show_metric = show_metric if show_metric is not None else self.plot_show_metric
        plot_data = self.data.loc[lambda dt:dt.Method.isin(show_metric)]
        
        plot = plot_alt.plot_TvP(
            data   = plot_data,
            y_name = self.y_name,
            scales = scales
        )
        return plot
    
    def plot_Pts(
        self,
        add_focus   :bool  = True,
        show_metric :list  = None,
        drop_anomaly:bool  = False,
        figure_size :tuple = (1200,None),
        scales      :str   = 'fixed'
    ):
        show_metric = show_metric if show_metric is not None else self.plot_show_metric
        plot_data = (
            self.data
            .loc[lambda dt:dt.Method.isin(show_metric)]
            .melt(id_vars=['Time','Method'],value_vars=[f'True_{self.y_name}','Pred'])
            .pivot(index=['Time','variable'],columns='Method',values='value')
            .sort_values('variable',ascending=False) # 调整图层顺序
            .reset_index()
        )
        plot = ts_line(
            data         = plot_data,
            x            = 'Time',
            y            = show_metric,
            color_by     = 'variable',
            add_focus    = add_focus,
            scales       = scales,
            fig_width    = figure_size[0],
            fig_height   = figure_size[1],
            color_legend = None
        )
        return plot
        
    def plot_Rts(
        self,
        add_focus   :bool   = True,
        show_metric :list   = None,
        add_iqr_line:bool   = False,
        figure_size  :tuple = (1200,None),
        scales      :str    = 'fixed',
    ):
        show_metric = show_metric if show_metric is not None else self.plot_show_metric
        
        plot_data = (
            self.data
            .loc[lambda dt:dt.Method.isin(show_metric)]
            .melt(id_vars=['Time','Method'],value_vars=['Resid'])
            .pivot(index=['Time','variable'],columns='Method',values='value')
            .reset_index()
        )
        
        plot = ts_line(
            data         = plot_data,
            x            = 'Time',
            y            = show_metric,
            add_focus    = add_focus,
            scales       = scales,
            fig_width    = figure_size[0],
            fig_height   = figure_size[1],
            color_legend = None
        )
        
        return plot
    
    def plot_Racf(self):
        ...
    
    def plot_Rar(self):
        ...
        #TODO 残差的自回归矩阵图
    
    def plot_Rqqnorm(self):
        ...
    
    def plot_RHeteroskedasticity(self):
        ...
        # sqrt_abs_std_resid