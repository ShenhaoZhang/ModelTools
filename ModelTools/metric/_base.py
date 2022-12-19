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
        
        self.y_true        = np.array(y_true)
        self.y_pred        = y_pred if isinstance(y_pred,list) else [y_pred]
        self.y_pred        = [np.array(pred) for pred in self.y_pred]
        self.__y_pred_name = y_pred_name if isinstance(y_pred_name,list) else [y_pred_name]
        self.sample_n      = len(self.y_true)
        self.y_pred_n      = len(self.y_pred)
        self.y_name        = y_name
        self.highlight     = {} if highlight is None else highlight
        self.index         = index
        self.index_freq    = index_freq
        self.__init_input()
        
        self.resid           = [self.y_true-pred for pred in self.y_pred]
        
        self.data = None
        self.__init_data()
    
    def __init_input(self):
        # 真实值与预测值的的长度校验
        for pred in self.y_pred:
            if len(pred) != len(self.y_true):
                raise ValueError('WRONG')
            if np.isnan(pred).any():
                raise ValueError('WRONG')
        # 预测值名称的长度校验
        if (self.__y_pred_name is not None) and (len(self.y_pred) != len(self.__y_pred_name)):
            raise Exception('Wrong')

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
            self.index = np.arange(self.sample_n)
        
        self.y_pred_name = []
        self.resid_name = []
        self.outlier_name = []
        for i,name in enumerate(self.__y_pred_name):
            if name is None:
                self.y_pred_name.append(f'Pred_{self.y_name}_{i}')
                self.resid_name.append(f'Resid_{self.y_name}_{i}')
                self.outlier_name.append(f'Outlier_{self.y_name}_{i}')
            else:
                self.y_pred_name.append(f'Pred_{name}')
                self.resid_name.append(f'Resid_{name}')
                self.outlier_name.append(f'Outlier_{name}')
    
        
    def __init_data(self,contain_resid=True):
        
        data      = [self.index, self.y_true, *self.y_pred]
        index     = ['Time',f'True_{self.y_name}', *self.y_pred_name]
        stubnames = ['Pred']
        if contain_resid:
            data      += [*self.resid]
            index     += [*self.resid_name]
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
    
    def get_metric(self,type='eval',style_metric=True,style_threshold=0.8,add_highlight_col=False):
        metric = self.metric(type)
        if add_highlight_col:
            metric = metric.assign(Highlight = lambda dt:dt.index.map(self.highlight)).fillna({'Highlight':'Others'})
        if style_metric:
            metric = self.style_metric(metric=metric,threshold=style_threshold)
        return metric
    
    def plot_metric_scatter(self,type='bv'):
        #TODO 处理异常大的值导致的可视化问题
        if type in ['bv','bv_robust']:
            metric = self.get_metric(type = 'resid', add_highlight_col = True,style_metric=False)
            plot = plot_alt.plot_metric_bias_var(data = metric, type = type)
        elif type == 'pca':
            metric = self.get_metric(type = 'eval', add_highlight_col = False,style_metric=False)
            plot = Pca(data = metric,scale = True).plot_bio(highlight = self.highlight)
        return plot
    
    #TODO 分块可视化metric变化的趋势
    def plot_metric_trend(self):
        ...
    
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