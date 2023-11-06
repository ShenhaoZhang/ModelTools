import itertools
from typing import Union

import numpy as np
import pandas as pd 
from scipy import stats
from .._src.tabulate import tabulate

class DataExplorer:
    
    def __init__(
        self,
        data     : Union[pd.DataFrame,dict],
        col_x    : Union[list,str,None]      = None,
        col_y    : Union[list,str,None]      = None,
        col_ts   : Union[str,None]           = None,
        col_group: Union[list,str,None]      = None
    ) -> None:
        self.data       = data.copy()
        self.data_n_row = self.data.shape[0] #FIXME data可能是字典
        self.data_n_col = self.data.shape[1]
        
        self.col_x      = col_x if col_x is not None else self.data.columns.to_list()
        self.col_x      = self.col_x if isinstance(self.col_x,list) else [self.col_x]
        
        self.col_y      = col_y if col_y is not None else []
        self.col_y      = self.col_y if isinstance(self.col_y,list) else [self.col_y]
        
        self.col_xy     = [*self.col_x,*self.col_y]
        self.col_ts     = col_ts
        self.col_group  = col_group if not isinstance(col_group,str) else [col_group]
        self._init_data()
        
        self.summary_data = {}
    
    def _init_data(self):
        if len(set(self.col_x).intersection(self.col_y)) > 0:
            raise Exception('col_x 与 col_y 有重叠')
        
        # 当data是由多个DataFrame构成的dict时，合并成一个DataFrame
        if isinstance(self.data,dict):
            data_list = []
            for label,df in self.data.items():
                df['_label_'] = label
                data_list.append(df)
            self.data = pd.concat(data_list,axis=0,ignore_index=True)
            self.is_labeled = True
        elif isinstance(self.data,pd.DataFrame):
            self.data['_label_'] = '_none_'
        
        # 当未输入col_ts参数时，用序号代替
        if self.col_ts is None:
            # TODO 检验col_ts的数据类型，不满足条件时，相同处理
            self.data = self.data.groupby('_label_',as_index=False).apply(lambda dt:dt.assign(_order_ = np.arange(len(dt))))
            self.col_ts = '_order_'
        
        # 从col_x中剔除其他角色的字段
        remove_from_col_xy = []
        for col in [self.col_ts,self.col_group]:
            if col is None:
                continue
            if isinstance(col,list):
                remove_from_col_xy += col 
            if isinstance(col,str):
                remove_from_col_xy.append(col)
        for col in remove_from_col_xy:
            if col in self.col_x:
                self.col_x.remove(col)
            if col in self.col_y:
                self.col_y.remove(col)
        
        # 数据类型分类
        all_col = [*self.col_x,*self.col_y]
        data = self.data.loc[:,all_col].head(1)
        self.numeric_col  = data.select_dtypes(include='number').columns.to_list()
        self.category_col = data.select_dtypes(include='category').columns.to_list()
        self.string_col   = data.select_dtypes(include='object').columns.to_list()
        
    def _get_col_info(self,data,col:str) -> dict:
        col_info = {'col':col}
        sr:pd.Series = data.loc[:,col]
        
        col_info['n_missing']     = sr.isna().sum()
        col_info['complete_rate'] = 1 - sr.isna().mean()
        
        if col in self.numeric_col:
            col_info['mean'] = sr.mean()
            col_info['sd']   = sr.std()
            col_info['skew'] = stats.skew(sr)
            col_info['kurt'] = stats.kurtosis(sr)
            for q in [0,0.25,0.5,0.75,1]:
                p = f'p{int(q*100)}'
                col_info[p] = sr.quantile(q=q)
            
        elif col in self.category_col:
            col_info['ordered'] = sr.cat.ordered
            col_info['n_unique'] = len(sr.cat.categories)
            
        elif col in self.string_col:
            col_info['len_min']    = sr.str.len().min()
            col_info['len_max']    = sr.str.len().max()
            col_info['n_unique']   = len(sr.unique())
            col_info['whitespace'] = (sr.str.strip(' ')==' ').sum()
            col_info['empty']      = (sr==' ').sum()
        
        return col_info
    
    def _get_groupby_col_info(self,data:pd.DataFrame,col:str,group:dict) -> dict:
        grp_col_info = {**group,'col':col}
        group_data = data
        for grp_col,grp_value in group.items():
            index = group_data.loc[:,grp_col] == grp_value
            group_data = group_data.loc[index,:]
        col_info = self._get_col_info(data=group_data,col=col)
        grp_col_info.update(col_info)
        return grp_col_info
    
    def summary(self):
        def print_table(table:pd.DataFrame,add_header=True):
            table = table.round(4)
            if add_header:
                tbl = tabulate(table,headers=table.columns,tablefmt='plain',showindex=False)
            else:
                tbl = tabulate(table,tablefmt='plain',showindex=False)
            print(tbl)
            
        print('─'*3 + ' Data Summary ' + '─'*20)
        table = {
            'n_rows' : self.data_n_row,
            'n_cols' : self.data_n_col,
            'numeric_cols' : len(self.numeric_col),
            'category_cols': len(self.category_col),
            'string_cols'  : len(self.string_col),
        }
        table = pd.Series(table).to_frame().reset_index()
        print_table(table,add_header=False)
        
        for type in ['numeric','category','string']:
            if len(eval(f'self.{type}_col')) == 0:
                continue
            print('─'*3 + f' Column type : {type} ' + '─'*20)
            table = []
            for col in eval(f'self.{type}_col'):
                if self.col_group is None:
                    table.append(self._get_col_info(data=self.data,col=col))
                elif self.col_group is not None:
                    iterrows = self.data.loc[:,self.col_group].drop_duplicates().iterrows()
                    for row in iterrows:
                        grp = row[1].to_dict()
                        table.append(self._get_groupby_col_info(data=self.data,col=col,group=grp))
            table = pd.DataFrame(table)
            self.summary_data[type] = table
            print_table(table)
        
    def plot_distribution(self):
        # 直方图
        # 密度图
        # 箱行图/小提琴图
        # 山脊图
        ...
    
    def plot_ts(self):
        ...
    
    # def plot_ts2_scatter(self,col_x:str,col_y:str,col_ts=None):
    #     plot = ts2_scatter(data=self.data,x=col_x,y=col_y,ts=col_ts)
    #     return plot
        
        
    # def plot_scatter_matrix(self):
    #     plot = matrix_scatter(self.data.loc[:,self.col_xy])
    #     return plot

if __name__ == '__main__':
    df = pd.DataFrame({
        'x1':np.random.normal(loc=2,scale=1,size=100),
        'x2':np.random.normal(loc=3,scale=2,size=100)
    })
    print(DataExplorer(df).summary())