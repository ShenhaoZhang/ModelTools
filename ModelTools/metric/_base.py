import numpy as np
import pandas as pd
from ..plot import plot_alt
from ..tools.pca import Pca

# warnings.filterwarnings("ignore")

class BaseMetric:
    """
    预测效果的评估指标
    """
    
    def __init__(
        self, 
        y_true     : np.ndarray,
        y_pred     : list,
        y_pred_name: list             = None,
        y_name     : str              = 'y',
        index      : pd.DatetimeIndex = None,
        index_freq : str              = None,
        highlight  : dict             = None
    ) -> None:
        self.y_true    = np.array(y_true)
        self.y_pred    = y_pred if isinstance(y_pred,list) else [y_pred]
        self.y_pred    = [np.array(pred) for pred in self.y_pred]
        self.sample_n  = len(self.y_true)
        self.y_pred_n  = len(self.y_pred)
        
        # 真实值与预测值的的长度校验
        for pred in self.y_pred:
            if len(pred) != len(y_true):
                raise ValueError('WRONG')
            if np.isnan(pred).any():
                raise ValueError('WRONG')
        # 预测值名称的长度校验
        self.y_pred_name = y_pred_name if isinstance(y_pred_name,list) else [y_pred_name]
        if (y_pred_name is not None) and (len(self.y_pred) != len(self.y_pred_name)):
            raise Exception('Wrong')
        self.highlight = {} if highlight is None else highlight
        
        self.y_name       = y_name
        self.y_pred_name  = [f'Pred_{y_name}_{i}' for i in range(self.y_pred_n)] if y_pred_name is None else [f'Pred_{name}' for name in y_pred_name]
        self.resid_name   = [f'Resid_{self.y_name}_{i}' for i in range(self.y_pred_n)] if y_pred_name is None else [f'Resid_{name}' for name in y_pred_name]
        self.outlier_name = [f'Outlier_{self.y_name}_{i}' for i in range(self.y_pred_n)] if y_pred_name is None else [f'Outlier_{name}' for name in y_pred_name]
        
        # 当索引不为None时的校验
        if index is not None:
            if not isinstance(index,pd.DatetimeIndex):
                try:
                    index = pd.DatetimeIndex(index)
                except Exception as E:
                    print(E)
                    raise TypeError('WRONG')
            if index_freq is None:
                raise ValueError('WRONG')
            if len(index) != len(y_true):
                raise ValueError('WRONG')
        self.index       = np.arange(self.sample_n) if index is None else index
        self.index_freq  = index_freq
        
        # 限定评估的数量
        highlight_count = len(self.highlight)
        if highlight_count >= 6:
            self.highlight_y = list(self.highlight.keys())
        else:
            top_metric_count = 6 - highlight_count
            top_metric = (
                self.get_metric()
                .drop(list(self.highlight.keys()))
                # .sort_values(by='MSE')
                .head(top_metric_count)
                .index.to_list()
            )
            self.highlight_y = list(self.highlight.keys()) + top_metric
        
        self.data = None
        self._init_outlier()
        self._init_data()
        
    def _init_outlier(self):
        pass
        
    def _init_data(self,contain_resid=True):
        
        data      = [self.index, self.y_true, *self.y_pred]
        index     = ['Time',f'True_{self.y_name}', *self.y_pred_name]
        stubnames = ['Pred']
        if contain_resid:
            data      += [*self.resid, *self.outlier_index]
            index     += [*self.resid_name, *self.outlier_name]
            stubnames += ['Resid','Outlier']
        
        self.data = (
            pd.DataFrame(data = data,index = index)
            .T
            .pipe(
                pd.wide_to_long,
                stubnames = stubnames,
                i         = 'Time',
                j         = 'Method',
                suffix    = '\w+',
                sep       = '_'
            )
            .reset_index()
            .infer_objects()
            .assign(Highlight = lambda dt:dt.Method.map(self.highlight))
        )
        
        # 时间索引填充空缺值
        if isinstance(self.index,pd.DatetimeIndex):
            time_start = self.data.Time.min()
            time_end = self.data.Time.max()
            complete_index = pd.MultiIndex.from_product(
                [pd.date_range(time_start,time_end,freq=self.index_freq), self.data.Method.drop_duplicates()],
                names=['Time','Method']
            )
            self.data = self.data.set_index(['Time','Method']).reindex(complete_index).reset_index()
        
    def get_metric(self):
        pass
    
    def plot_metric_scatter(self,type='bv'):
        
        #TODO 处理异常大的值导致的可视化问题
        if type in ['bv','bv_robust']:
            metric = self.get_metric(type = 'resid', add_highlight_col = True)
            plot = plot_alt.plot_metric_bias_var(data = metric, type = type)
        elif type == 'pca':
            metric = self.get_metric(type = 'eval', add_highlight_col = False)
            plot = Pca(data = metric,scale = True).plot_bio(highlight = self.highlight)
            
        return plot
    
    #TODO 分块可视化metric变化的趋势
    def plot_metric_trend(self):
        ...
    