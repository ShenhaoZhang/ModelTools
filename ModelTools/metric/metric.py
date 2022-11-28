import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from tabulate import tabulate

from . import plot_gg
from . import plot_alt
from ..tools.pca import Pca
from ..plot.evo import multi_line

# warnings.filterwarnings("ignore")

class Metric:
    """
    预测效果的评估指标
    """
    
    def __init__(self, y_true:np.ndarray, y_pred:list, y_pred_name:list = None, y_name = 'y', 
                 index:pd.DatetimeIndex = None, index_freq:str = None, highlight:dict = None) -> None:
        self.y_true    = np.array(y_true)
        self.y_pred    = y_pred if isinstance(y_pred,list) else [y_pred]
        self.y_pred    = [np.array(pred) for pred in self.y_pred]
        self.sample_n  = len(self.y_true)
        self.y_pred_n  = len(self.y_pred)
        
        # 真实值与预测值的的长度校验
        for pred in self.y_pred:
            if len(pred) != len(y_true):
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
            self.focus = list(self.highlight.keys())
        else:
            top_metric_count = 6 - highlight_count
            top_metric = (
                self.get_metric()
                .drop(list(self.highlight.keys()))
                .sort_values(by='MSE')
                .head(top_metric_count)
                .index.to_list()
            )
            self.focus = list(self.highlight.keys()) + top_metric
        
        self.data = None
        self._init_data()
        
    def _init_data(self):
        
        # 残差及残差中的异常值
        self.resid           = [self.y_true-pred for pred in self.y_pred]
        resid_total          = np.array(self.resid).flatten()
        self.resid_total_iqr = np.quantile(resid_total,q=0.75) - np.quantile(resid_total,q=0.25)
        # 异常值的定义是残差大于1.5倍IQR的预测样本点
        self.outlier_index = [np.abs(self.resid[i]) > (1.5 * self.resid_total_iqr) for i in range(self.y_pred_n)]
        self.outlier_count = [np.sum(self.outlier_index[i]) for i in range(self.y_pred_n)]
        self.outlier_pct   = [np.round(self.outlier_count[i] / self.sample_n,4) for i in range(self.y_pred_n)]
        
        self.data = (
            pd.DataFrame(
                data=[self.index, self.y_true, *self.y_pred, *self.resid, *self.outlier_index],
                index=['Time',f'True_{self.y_name}', *self.y_pred_name, *self.resid_name, *self.outlier_name]
            )
            .T
            .pipe(pd.wide_to_long,stubnames=['Pred','Resid','Outlier'],i='Time',j='Method',suffix='\w+',sep='_')
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
        
    def get_metric(self,type='eval',add_highlight_col=False):
        
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
            metric_dict['resid_Outlier_pct'] = self.outlier_pct
        else:
            raise TypeError('WRONG')
        
        metric = pd.DataFrame(metric_dict,index=[name.strip('Pred_') for name in self.y_pred_name]).sort_index()
        if add_highlight_col is True:
            metric = metric.assign(Highlight = lambda dt:dt.index.map(self.highlight)).fillna({'Highlight':'Others'})
        
        return metric
    
    def _get_plot_caption(self,type):
        metric = self.get_metric(type=type,add_highlight_col=False)
        if type == 'eval':
            caption = (
                f'Sample_N={self.sample_n}\n' + 
                tabulate(
                    metric
                    .applymap(lambda x:'%.3f'%x )
                    .apply(lambda sr:sr.str.ljust(sr.str.strip('-').str.len().max(),'0'))
                    .apply(lambda sr:metric.columns.str.cat(sr,sep=':'), axis=1, result_type='expand')
                    .reset_index(),
                    tablefmt='plain',
                    showindex=False
                )
            )
        
        if type == 'resid':
            caption = (
                f'Sample_N={self.sample_n}\n' + 
                tabulate(
                    metric
                    .applymap(lambda x:'%.3f'%x )
                    .apply(lambda sr:sr.str.ljust(sr.str.strip('-').str.len().max(),'0'))
                    .apply(lambda sr:metric.columns.str
                        .extract(r'(?<=resid_)(.+)',expand=False).dropna().str.cat(sr.values,sep=':'),
                        axis=1,
                        result_type='expand')
                    .reset_index(),
                    tablefmt='plain',
                    showindex=False
                )
            )
        return caption
    
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
    
    #TODO 可选日期作为y轴
    def plot_TvP(self,focus:list = None, add_lm=False, add_outlier=False, add_quantile=False, figure_size=(10, 5), scales='fixed', engine='gg'):
        focus = focus if focus is not None else self.focus
        plot_data = self.data.loc[lambda dt:dt.Method.isin(focus)]
        
        caption = self._get_plot_caption(type='eval')
        if engine == 'gg':
            plot = plot_gg.gg_Tvp(
                data         = plot_data,
                y_name       = self.y_name,
                y_pred_name  = self.y_pred_name,
                caption      = caption,
                add_lm       = add_lm,
                add_outlier  = add_outlier,
                add_quantile = add_quantile,
                figure_size  = figure_size,
                scales       = scales
            )
        elif engine == 'alt':
            plot = plot_alt.plot_TvP(
                data   = plot_data,
                y_name = self.y_name,
                scales = scales
            )
        return plot
    
    def plot_Pts(self,focus:list = None,time_limit=None,drop_anomaly=False,figure_size=(10, 5),scales='fixed',engine='gg'):
        focus = focus if focus is not None else self.focus
        plot_data = self.data.loc[lambda dt:dt.Method.isin(focus)]
        
        caption = self._get_plot_caption(type='eval')
        if engine == 'gg':
            plot = plot_gg.gg_Pts(
                data         = plot_data,
                y_name       = self.y_name,
                caption      = caption,
                time_limit   = time_limit,
                drop_anomaly = drop_anomaly,
                figure_size  = figure_size,
                scales       = scales
            )
        elif engine == 'alt':
            plot = multi_line(
                data  = plot_data,
                x     = 'Time',
                y     = 'Pred',
                facet = 'Method'
            )
        return plot
        
    def plot_Rts(self,focus:list = None,add_iqr_line=False,time_limit=None,figure_size=(10, 5),scales='fixed',engine='gg'):
        focus = focus if focus is not None else self.focus
        plot_data = self.data.loc[lambda dt:dt.Method.isin(focus)]
        
        caption = self._get_plot_caption(type='resid')
        if engine == 'gg':
            plot = plot_gg.gg_Rts(
                data         = plot_data,
                caption      = caption,
                add_iqr_line = add_iqr_line,
                iqr          = self.resid_total_iqr,
                time_limit   = time_limit,
                figure_size  = figure_size,
                scales       = scales
            )
        elif engine == 'alt':
            plot = multi_line(
                data   = plot_data,
                x      = 'Time',
                y      = 'Resid',
                facet  = 'Method',
                scales = scales
            )
        return plot
    
    def plot_Rar(self):
        ...
        #TODO 残差的自回归矩阵图


if __name__ == '__main__':
    n = 1000
    x = np.random.randn(n)*2
    y1 = x+np.random.randn(n)
    y2 = x+2*np.random.randn(n)
    y3 = x+3*np.random.randn(n)
    em = Metric(x,[y1,y2,y3])
    print(em.get_metric())
    print(em.data)
    print(em.plot_TvP(add_quantile=True))