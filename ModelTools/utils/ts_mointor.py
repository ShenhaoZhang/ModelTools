import numpy as np 
import pandas as pd 
import plotnine as gg

class Monitor:
    def __init__(
        self,
        data   : pd.DataFrame,
        col_ts : str,
        col_var: list,
    ) -> None:
        
        self.data    = data
        self.col_ts  = col_ts
        self.col_var = col_var if isinstance(col_var,list) else [col_var]
        
        self.all_point     = None
        self.cur_point     = None
        self.cur_point_idx = None
        
        self.set_var    = {}
        self.agg_method = np.median
    
    def filter_all_cur_diff(
        self, 
        cols      : list,
        value     : float,
        lim_freq  : int = 1,
        win_radius: int = 5
    ):
        # 寻找一个出现指标变化的时间点，且时间点前后该指标不会有其他变化
        # 筛选col的变化值大于value的时间，并满足在win_radius窗口内出现小于等于lim_freq次
        
        # lim_freq 限制窗口期内出现的次数
        cols = [cols] if isinstance(cols,str) else cols
        for col in cols:
            is_dyn_point = self.data.loc[:,col].diff().abs() >= value
            is_freq_lim  = is_dyn_point.rolling(
                window      = win_radius * 2,
                center      = True,
                min_periods = win_radius * 2,
                closed      = 'both'
            ).sum() <= lim_freq
            self.set_var[f'current_diff_{col}'] = is_dyn_point & is_freq_lim
        return self 
    
    def filter_all_bef_stable(
        self,
        cols         : list,
        method       : callable = np.ptp,
        threshold    : float = 1e6,
        before_period: int = 5
    ):
        # 寻找一个时间点，该时间点前某个指标变化稳定
        cols = [cols] if isinstance(cols,str) else cols
        for col in cols:
            method_value = (
                self.data.loc[:,col]
                .rolling(
                    window      = before_period,
                    center      = False,
                    min_periods = before_period,
                    closed      = 'both'
                )
                .apply(method)
            )
            self.set_var[f'before_static_{col}'] = method_value <= threshold
        return self
    
    def find_point(self):
        is_point = np.array([True]*len(self.data))
        for col_res in self.set_var.values():
            is_point &= col_res.values  
        
        self.all_point = self.data.loc[is_point,self.col_ts]
        
        if len(self.all_point) == 0:
            raise Exception('未找到Point')
        else:
            print(f'发现{len(self.all_point)}个Point')
        
        self.select_point(idx=0)
        return self
    
    def select_point(self,idx=0):
        if idx > len(self.all_point)-1:
            raise Exception('超出Point数量的范围')
        self.cur_point = self.all_point.iloc[idx]
        self.cur_point_idx = idx 
        return self
    
    def _split_data_by_point(
        self,
        before_period:int = 10,
        after_period :int = 10,
        unit         :str = 'minutes'
    ) -> tuple:
        before_start = self.cur_point - pd.Timedelta(value=before_period,unit=unit)
        after_end    = self.cur_point + pd.Timedelta(value=after_period,unit=unit)
        
        data_before = self.data.loc[self.data[self.col_ts].between(before_start,self.cur_point,inclusive='left')]
        data_after  = self.data.loc[self.data[self.col_ts].between(self.cur_point,after_end)]
        return data_before,data_after
    
    def plot_all_point(self):
        gg.options.figure_size = [10,len(self.col_var)*3]
        n_point = len(self.all_point)
        
        plot = (
            self.data
            .melt(id_vars=self.col_ts,value_vars=self.col_var)
            .pipe(gg.ggplot)
            + gg.aes(x='ts',y='value')
            + gg.geom_line()
            + gg.facet_wrap('variable',scales='free_y',ncol=1)
            + gg.geom_vline(xintercept=self.all_point,color='red')
            + gg.labs(title=f'Find {n_point} Points')
        )
        return plot
    
    def _predict_data(
        self,
        data_train: pd.DataFrame,
        data_pred : pd.DataFrame,
        pred_fun  : dict
    ) -> pd.DataFrame:
        
        ts = data_pred.loc[:,self.col_ts] 
        
        all_pred_data = []
        for col_name,col_fun in pred_fun.items():
            for FUN,ARGS in col_fun.items():
                if ARGS is None:
                    ARGS = {}
                pred = FUN(data_train,data_pred,**ARGS)
                pred_data = pd.DataFrame(
                    data = {
                        'type'     : 'after',
                        'variable' : col_name,
                        'pred'     : pred,
                        self.col_ts: ts
                    }
                )
                all_pred_data.append(pred_data)
        
        all_pred_data = pd.concat(all_pred_data,axis=0)
        
        return all_pred_data
    
    def plot_select_point(
        self,
        before_period:int  = 10,
        after_period :int  = 10,
        plot_var     :list = None,
        pred_fun     :dict = None,
        select_next  :bool = False
    ):
        gg.options.figure_size = [10,len(self.col_var)*3]
        
        split_data  = self._split_data_by_point(before_period,after_period)
        data_before = split_data[0].assign(type='before')
        data_after  = split_data[1].assign(type='after' )
        
        plot_data   = (
            pd.concat([data_before,data_after],axis=0)
            .melt(
                id_vars    = [self.col_ts,'type'],
                value_vars = self.col_var,
                var_name   = 'variable',
                value_name = 'value'
            )
            .groupby(['variable','type'],as_index=False)
            .apply(lambda dt:dt.assign(value_agg=self.agg_method(dt['value'])))
        )
        
        if plot_var is not None:
            plot_data = plot_data.loc[lambda dt:dt.variable.isin(plot_var)]
        
        if pred_fun is not None:
            pred_data = self._predict_data(data_before,data_after,pred_fun)
            plot_data = plot_data.merge(pred_data,how='left',on=['ts','type','variable'])
        
        plot = (
            gg.ggplot(plot_data)
            + gg.aes(x='ts')
            + gg.geom_line(gg.aes(y='value'))
            + gg.geom_point(gg.aes(y='value'))
            + gg.geom_line(gg.aes(y='value_agg',group='type'),color='blue',linetype='--')
            + gg.geom_vline(xintercept=self.cur_point,color='red',linetype='--')
            + gg.geom_label(
                gg.aes(x='ts',y='value_agg',label='value_agg'),
                data=lambda dt:
                    dt.groupby(['type','variable'],as_index=False)
                    .apply(lambda dt:dt.head(1) if dt['type'].iat[0]=='before' else dt.tail(1))
                    .assign(value_agg=lambda dt:dt.value_agg.round(2)),
                color='blue',
                label_padding=0.15
            )
            + gg.facet_wrap('variable',scales='free_y',ncol=1)
            + gg.scale_x_datetime(date_labels='%m/%d %H:%M')
            + gg.labs(title=f'Current Point ({self.cur_point_idx+1}/{len(self.all_point)}) : {self.cur_point}')
        )
        
        if pred_fun is not None:
            plot = (
                plot 
                + gg.geom_point(gg.aes(y='pred'),color='green')
                + gg.geom_line(gg.aes(y='pred'),color='green',linetype='--')
            )
        
        if select_next:
            next_idx = self.cur_point_idx + 1
            if next_idx >= len(self.all_point):
                next_idx = 0
            self.select_point(idx=next_idx)
            
        return plot