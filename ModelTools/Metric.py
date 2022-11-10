import warnings

import numpy as np
import pandas as pd
import plotnine as gg
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from tabulate import tabulate

warnings.filterwarnings("ignore")

#TODO 增加图形的交互功能

class Metric:
    """
    预测值的效果监控
    """
    
    def __init__(self, y_true:np.ndarray, y_pred:list, y_pred_name:list = None, y_name = 'y', index:pd.DatetimeIndex = None, index_freq:str = None) -> None:
        self.y_true     = y_true
        self.y_pred     = y_pred if isinstance(y_pred,list) else [y_pred]
        
        self.sample_n    = len(self.y_true)
        self.y_pred_n    = len(self.y_pred)
        self.y_pred_name = y_pred_name if isinstance(y_pred_name,list) else [y_pred_name]
        if (y_pred_name is not None) and (len(self.y_pred) != len(self.y_pred_name)):
            raise Exception('Wrong')
        self.y_name       = y_name
        self.y_pred_name  = [f'Pred_{y_name}_{i}' for i in range(self.y_pred_n)] if y_pred_name is None else [f'Pred_{name}' for name in y_pred_name]
        self.resid_name   = [f'Resid_{self.y_name}_{i}' for i in range(self.y_pred_n)] if y_pred_name is None else [f'Resid_{name}' for name in y_pred_name]
        self.outlier_name = [f'Outlier_{self.y_name}_{i}' for i in range(self.y_pred_n)] if y_pred_name is None else [f'Outlier_{name}' for name in y_pred_name]
        
        if index is not None:
            if not isinstance(index,pd.DatetimeIndex):
                raise TypeError('WRONG')
            if index_freq is None:
                raise TypeError('WRONG')
        self.index       = np.arange(self.sample_n) if index is None else index
        self.index_freq  = index_freq
        
        self.data = None
        self._init_data()
        self._init_plot_caption()
        
        print(self.get_metric())
        
    def _init_data(self):
        
        # 残差
        self.resid        = [self.y_true-pred for pred in self.y_pred]
        self.resid_mean   = [np.mean(resid) for resid in self.resid]
        self.resid_sd     = [np.std(resid) for resid in self.resid]
        self.resid_skew   = [stats.skew(resid) for resid in self.resid]
        self.resid_kurt   = [stats.kurtosis(resid) for resid in self.resid]
        resid_total       = np.array(self.resid).flatten()
        self.residual_iqr = np.quantile(resid_total,q=0.75) - np.quantile(resid_total,q=0.25)
        
        # 异常值的定义是残差大于1.5倍IQR的预测样本点
        self.outlier_index = [np.abs(self.resid[i]) > (1.5 * self.residual_iqr) for i in range(self.y_pred_n)]
        self.outlier_count = [np.sum(self.outlier_index[i]) for i in range(self.y_pred_n)]
        self.outlier_pct   = [np.round(self.outlier_count[i] / self.sample_n,4) for i in range(self.y_pred_n)]
        
        # 评价指标
        self.r2   = [metrics.r2_score(self.y_true,self.y_pred[i]) for i in range(self.y_pred_n)]
        self.mae  = [metrics.mean_absolute_error(self.y_true,self.y_pred[i]) for i in range(self.y_pred_n)]
        self.mape = [metrics.mean_absolute_percentage_error(self.y_true,self.y_pred[i]) for i in range(self.y_pred_n)]
        #TODO MBE MSE MAPE MdAE std_MAE(iqr) std_MAPE(iqr)
        
        self.data = (
            pd.DataFrame(
                data=[self.index, self.y_true, *self.y_pred, *self.resid, *self.outlier_index],
                index=['Time',f'True_{self.y_name}', *self.y_pred_name, *self.resid_name, *self.outlier_name]
            )
            .T
            .pipe(pd.wide_to_long,stubnames=['Pred','Resid','Outlier'],i='Time',j='method',suffix='\w+',sep='_')
            .reset_index()
            .infer_objects()
            .assign(method=lambda dt:'Pred_'+dt.method)
        )
        
        if isinstance(self.index,pd.DatetimeIndex):
            time_start = self.data.Time.min()
            time_end = self.data.Time.max()
            complete_index = pd.MultiIndex.from_product(
                [pd.date_range(time_start,time_end,freq=self.index_freq), self.data.method.drop_duplicates()],
                names=['Time','method']
            )
            self.data = (
                self.data 
                .set_index(['Time','method'])
                .reindex(complete_index)
                .reset_index()
            )
    
    def _init_plot_caption(self):
        metric = self.get_metric().loc[:,['R2','MAE','MAPE']]
        self.caption = (
            f'Sample_N={self.sample_n}\n' + 
            tabulate(
                metric
                .applymap(lambda x:'%.3f'%x)
                .apply(lambda sr:sr.str.strip('-').str.ljust(sr.str.len().max(),'0'))
                .apply(lambda sr:metric.columns.str.cat(sr,sep=':'), axis=1, result_type='expand')
                .reset_index(),
                tablefmt='plain',
                showindex=False
            )
        )
        
        metric_resid = self.get_metric().loc[:,lambda dt:dt.columns.str.contains('resid_')]
        self.caption_resid = (
            f'Sample_N={self.sample_n}\n' + 
            tabulate(
                metric_resid
                .applymap(lambda x:'%.3f'%x)
                .apply(lambda sr:sr.str.strip('-').str.ljust(sr.str.len().max(),'0'))
                .apply(lambda sr:metric_resid.columns.str
                       .extract(r'(?<=resid_)(.+)',expand=False).dropna().str.cat(sr.values,sep=':'),
                       axis=1,
                       result_type='expand')
                .reset_index(),
                tablefmt='plain',
                showindex=False
            )
        )
        
    def get_metric(self):
        #TODO 增加ACF特征
        metric = pd.DataFrame({
            'R2'               : self.r2,
            'MAE'              : self.mae,
            'MAPE'             : self.mape,
            'resid_Mean'       : self.resid_mean,
            'resid_SD'         : self.resid_sd,
            'resid_Skew'       : self.resid_skew,
            'resid_Kurt'       : self.resid_kurt,
            'resid_IQR'        : self.residual_iqr,
            'resid_Outlier_pct': self.outlier_pct
        },index=self.y_pred_name)
        
        return metric
    
    def plot_metric_TVP(self,add_lm=False,add_outlier=False,add_quantile=False,figure_size=(6.4, 4.8)):
        gg.options.figure_size = figure_size
        plot = (
            gg.ggplot(self.data)
            + gg.aes(x=f'True_{self.y_name}',y='Pred')
            + gg.geom_point()
            + gg.facet_wrap(facets='method')
            + gg.labs(
                title=f'True_{self.y_name} VS Predict_{self.y_name}',
                caption=self.caption,
                x = f'True_{self.y_name}',
                y = f'Pred_{self.y_name}'
            )
        )
        
        if add_lm:
            plot += gg.geom_smooth(method='lm',color='blue')
            
        if add_outlier:
            plot = plot + gg.geom_point(gg.aes(color='Outlier')) + gg.scale_color_manual(values=['black','red'])
            
        if add_quantile:
            for q in [0.05,0.95]:
                mod = Pipeline([
                    ('std',StandardScaler()),
                    ('spline',PolynomialFeatures()),
                    ('qr',QuantileRegressor(quantile=q,alpha=0,solver='highs')),
                ])
                x_sample, y_hat, method_sample = np.array([]), np.array([]) ,np.array([])
                for name in self.y_pred_name:
                    data_name = self.data.loc[lambda dt:dt.method==name,:]
                    x = data_name.loc[:,f'True_{self.y_name}'].to_numpy()
                    y = data_name.loc[:,'Pred'].to_numpy()
                    mod.fit(x.reshape(-1,1),y)
                    y_hat = np.append(y_hat,mod.predict(X=x.reshape(-1,1)))
                    x_sample = np.append(x_sample,x)
                    method_sample = np.append(method_sample,[name]*len(x))
                    
                plot += gg.geom_line(
                    data=pd.DataFrame({'x':x_sample,'y':y_hat,'method':method_sample}),
                    mapping = gg.aes(x='x',y='y'),
                    color='red',
                    linetype='--'
                )
        plot += gg.geom_abline(slope=1,intercept=0,color='red',size=1)
                
        return plot
    
    def plot_metric_TSP(self,time_limit=None,figure_size=(10, 5)):
        gg.options.figure_size = figure_size
        plot = (
            gg.ggplot(pd.DataFrame(self.data))
            + gg.aes(x = 'Time')
            + gg.geom_line(gg.aes(y=f'True_{self.y_name}',color="'True'"))
            + gg.geom_line(gg.aes(y='Pred',color="'Pred'"),data=self.data)
            + gg.facet_wrap(facets='method',ncol=1)
            + gg.scale_color_manual(values=['black','green'])
            + gg.labs(
                color = ' ',
                title = f'Time Series for True_{self.y_name} and Predict_{self.y_name}',
                caption = self.caption,
                y = self.y_name
            )
        )
        
        if time_limit:
            plot += gg.scale_x_continuous(limits=time_limit)
            #TODO 增加时间index的处理
        
        return plot
        
    def plot_metric_TSR(self,add_iqr_line=False,time_limit=None,figure_size=(10, 5)):
        gg.options.figure_size = figure_size
        plot = (
            gg.ggplot(self.data)
            + gg.geom_line(gg.aes(x='Time',y='Resid'))
            + gg.geom_hline(yintercept=0,size=1,color='red')
            + gg.facet_wrap(facets='method',ncol=1)
            + gg.labs(
                title = 'Time Series for Residual',
                caption=self.caption_resid,
            )
        )
        
        # TODO增加指示不同预测结果相差较大的区域
        
        if add_iqr_line:
            plot = (
                plot 
                + gg.geom_hline(gg.aes(linetype='"+- 1.5 * IQR"',yintercept=self.residual_iqr * 1.5),color='green',size=0.5)
                + gg.geom_hline(gg.aes(linetype='"+- 1.5 * IQR"',yintercept=-self.residual_iqr * 1.5),color='green',size=0.5)
                + gg.scale_linetype_manual(name=' ',values=['--','--'])
            )
        
        if time_limit:
            plot += gg.scale_x_continuous(limits=time_limit)
            #TODO 增加时间index的处理
        
        return plot
    
if __name__ == '__main__':
    n = 1000
    x = np.random.randn(n)*2
    y1 = x+np.random.randn(n)
    y2 = x+2*np.random.randn(n)
    y3 = x+3*np.random.randn(n)
    em = Metric(x,[y1,y2,y3])
    print(em.get_metric())
    print(em.data)
    print(em.plot_metric_TVP(add_quantile=True))