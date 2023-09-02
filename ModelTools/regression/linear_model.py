import re
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from patsy import dmatrices,build_design_matrices

from ..utils.data_grid import DataGrid
from .._src.tabulate import tabulate
from .metric import Metric
from .plot.distribution import plot_distribution
from .plot.grid import plot_grid,plot_all_grid
from .plot.check_model import plot_check_model
from .model_config import (
    _linear_model,
    _default_param,
    _cv_param_grid
)

class LinearModel:
    
    def __init__(
        self,
        formula      : str,
        data         : Union[dict,pd.DataFrame],
        show_progress: bool  = True,
        rng_seed     : float = 0
    ) -> None:
        
        self.formula       = formula
        self.formula_x     = re.findall('~(.+)',self.formula)[0]
        self.y_col         = re.findall('(.+)~',self.formula)[0]
        self.data          = self._init_data(new_data=data) 
        self.x,self.y      = self._model_dataframe(data=self.data,formula=self.formula)
        self.mod           = None
        self.coef_dist     = None
        self.show_progress = show_progress
        self.rng_seed      = rng_seed
        
        
    def _init_data(self,new_data=None,data_grid=None):
        # TODO 检查datagrid的合规性
        if new_data is not None and data_grid is not None:
            raise Exception('WRONG')
        
        elif new_data is None and data_grid is None:
            data = self.data
        
        elif new_data is None and data_grid is not None:
            data  = DataGrid(self.data.drop(self.y_col,axis=1)).get_grid(**data_grid)
        
        elif new_data is not None and data_grid is None:
            if isinstance(new_data,pd.DataFrame):
                data = new_data.reset_index(drop=True)
            elif isinstance(new_data,(dict,list,tuple)) :
                data = pd.DataFrame(new_data)
            
        return data
        
        
    def _model_dataframe(self,data,formula) -> Union[tuple,pd.DataFrame]:
        
        def matrix_to_df(matrix):
            data = np.asarray(matrix)
            df   = pd.DataFrame(data=data,columns=matrix.design_info.column_names)
            return df
        
        if '~' in formula:
            matrices = dmatrices(formula,data)
            # 保存训练模型时数据的design_info，用于转换新数据
            self._matrices_design_info = matrices[1].design_info
            y = np.asarray(matrices[0]).flatten()
            x = matrix_to_df(matrices[1])
            return x,y
        else:
            matrix = build_design_matrices([self._matrices_design_info], data)[0]
            x      = matrix_to_df(matrix)
            return x
    
    def fit(self,method='OLS',method_kwargs:dict=None,n_bootstrap=1000):
        
        # 模型默认参数
        mod_default_param = _default_param.get(method,{})
        # 模型指定参数
        method_kwargs = {} if method_kwargs is None else method_kwargs
        # 合并参数
        mod_default_param.update(method_kwargs)
                
        mod = _linear_model[method](fit_intercept=False,**mod_default_param)
        if method in _cv_param_grid.keys():
            
            # 从待交叉验证的参数中移除已指定的参数
            cv_param_grid:dict = _cv_param_grid.copy()[method]
            for method_kw in mod_default_param.keys():
                if method_kw in cv_param_grid.keys():
                    cv_param_grid.pop(method_kw)
            
            # 交叉验证寻找超参数
            cv = GridSearchCV(
                estimator  = mod,
                param_grid = cv_param_grid,
                n_jobs     = -1,
                refit      = True,
                cv         = 5
            )
            self.mod = cv.fit(X=self.x,y=self.y).best_estimator_
        else:
            self.mod = mod.fit(X=self.x, y=self.y)
        
        self.fit_resid = self.y - self.mod.predict(self.x)
        self.coef_dist = self.bootstrap_coef(n_bootstrap=n_bootstrap,re_boot=True)
        
        return self

    def bootstrap_coef(self,n_bootstrap:int=1000,re_boot:bool=False) -> pd.DataFrame:
        if n_bootstrap <= 0:
            return None
        
        self._check_fitted()
        
        rng       = np.random.default_rng(seed=self.rng_seed)
        all_index = np.arange(self.x.shape[0])
        data_size = len(all_index)
        x         = self.x.to_numpy()
        mod       = clone(self.mod)
        
        coef_dist = []
        for _ in get_progress_bar(range(n_bootstrap),self.show_progress):
            sample_index = rng.choice(all_index,size=data_size,replace=True)
            sample_x     = x[sample_index,:]
            sample_y     = self.y[sample_index]
            mod.fit(sample_x,sample_y)
            coef_dist.append(mod.coef_)
            
        coef_dist = pd.DataFrame(data = coef_dist,columns = self.x.columns)
        
        return coef_dist

    def bootstrap_pred(self,new_x=None,n_bootstrap=None) -> np.ndarray:
        new_x = self.x if new_x is None else new_x
        
        coef_dist       = self.coef_dist.to_numpy()
        n_coef_resample = coef_dist.shape[0]
        if (n_bootstrap is not None) and (n_coef_resample >= n_bootstrap):
            coef_dist = coef_dist[0:n_bootstrap,:]
        #TODO
        mod = clone(self.mod).fit(self.x.iloc[0:10,:],self.y[0:10])
        pred_dist = []    
        for coef in get_progress_bar(coef_dist,self.show_progress):
            mod.coef_ = coef
            pred      = mod.predict(new_x)
            pred_dist.append(pred)
        pred_dist = np.row_stack(pred_dist)
        
        return pred_dist
    
    def coef(self, hypothesis:float=0, alternative='two_side', ci_level=0.95) -> pd.DataFrame:
        self._check_fitted()
        
        coef_name  = self.x.columns
        coef_value = self.mod.coef_.flatten()
        coef       = pd.DataFrame(data={'estimate':coef_value},index=coef_name)
        
        # 用bootstrap方法对参数进行统计推断
        if self.coef_dist is not None:
            low_level = (1 - ci_level) / 2
            up_level  = ci_level + (1 - ci_level) / 2
            ci_lower  = np.quantile(self.coef_dist,low_level,axis=0)
            ci_upper  = np.quantile(self.coef_dist,up_level,axis=0)
            std_error = np.std(self.coef_dist,axis=0)
            z_score   = (coef.estimate - hypothesis) / std_error
            
            p_value = []
            for name in coef_name:
                value = get_p_value(
                    self.coef_dist.loc[:,name].to_numpy(),
                    hypothesis  = hypothesis,
                    alternative = alternative,
                )
                p_value.append(value)
            
            summary = pd.DataFrame(
                {
                    'std_Error': std_error,
                    'z'        : z_score,
                    'p_value'  : p_value,
                    'ci_lower' : ci_lower,
                    'ci_upper' : ci_upper
                },
                index = coef_name
            )
            
            coef = pd.concat([coef,summary],axis=1).loc[:,['estimate','std_Error','z','p_value','ci_lower','ci_upper']]
            
        return coef
    
    def plot_coef_dist(self,**plot_kwargs):
        plot = plot_distribution(
            data = self.coef_dist,
            **plot_kwargs
        )
        return plot
    
    def plot_coef_pair(self):
        ...
    
    def metric(self) -> pd.DataFrame:
        metric = Metric(y_true=self.y,y_pred=self.mod.predict(self.x)).get_metric()
        return metric
    
    def plot_metric(self):
        ...
        
    def summary(self):
        coef_info   = self.coef()
        metric_info = self.metric()
        print(
            tabulate(coef_info,headers='keys')
        )
        print(
            tabulate(metric_info,headers='keys')
        )
    
    def plot_check(self,ppc_n_resample=50):
        rng       = np.random.default_rng(self.rng_seed)
        pred      = self._predict(new_data=self.data)
        boot_pred = self.bootstrap_pred(n_bootstrap=ppc_n_resample)
        boot_pred += rng.choice(self.fit_resid,size=boot_pred.shape,replace=True)
        plot = plot_check_model(
            residual     = self.fit_resid,
            fitted_value = pred,
            boot_pred    = boot_pred,
            y_name       = self.y_col
        )
        return plot
    
    def slope(
        self,
        new_data  : pd.DataFrame = None,
        data_grid : dict         = None,
        slope_var : list         = None,
        ci_level  : float        = 0.95,
        eps       : float        = 1e-4
    ) -> pd.DataFrame:

        data  = self._init_data(new_data,data_grid)
        x     = self._model_dataframe(data,formula=self.formula_x)
        alpha = 1 - ci_level
        pred  = self.bootstrap_pred(x)
        
        # 指定斜率对应的变量
        if isinstance(slope_var,str):
            slope_var = [slope_var]
        elif slope_var is None:
            slope_var = data.columns.to_list()
        elif not isinstance(slope_var,list):
            raise Exception('slope_var必须是list')
        
        result_data = []
        for var_name in slope_var:
            
            # 计算slope的分布
            data_eps            = data.copy(deep=True)
            data_eps[var_name] += eps
            x_eps               = self._model_dataframe(data_eps,formula=self.formula_x)
            pred_eps            = self.bootstrap_pred(x_eps)
            slope_dist          = (pred_eps - pred) / eps
            
            # 基于完整数据计算的slope
            pred_slope_mean = (self._predict(new_x=x_eps) - self._predict(new_x=x)) / eps
            
            # 聚合结果 
            slope_result = pd.DataFrame(
                {
                    'term'         : var_name,
                    'mean'         : pred_slope_mean,
                    'mean_se'      : slope_dist.std(axis=0),
                    'mean_ci_lower': np.quantile(slope_dist,alpha/2,axis=0),
                    'mean_ci_upper': np.quantile(slope_dist,1-alpha/2,axis=0),
                }
            )
            slope_result = pd.concat([slope_result,data],axis=1)
            result_data.append(slope_result)
        
        result_data = pd.concat(result_data,axis=0)
        return result_data
    
    def plot_slope(
        self,
        data_grid:dict,
        **slope_kwargs
    ):
        slope_var = list(data_grid.keys())[0]
        slope_kwargs.update({'data_grid':data_grid,'slope_var':slope_var})
        slope = self.slope(**slope_kwargs)
        plot = plot_grid(
            data     = slope,
            plot_var = list(data_grid.keys()),
            ci_type  = 'mean',
            h_line   = 0,
            y_label  = self.y_col
        )
        return plot
    
    def compare_slope(self):
        ...
    
    def prediction(
        self, 
        new_data  : pd.DataFrame  = None,
        data_grid : dict          = None,
        ci_level  : float         = 0.95,
        ci_method : str           = 'bootstrap'
    ) -> pd.DataFrame:
        
        self._check_fitted()
        data  = self._init_data(new_data,data_grid)
        new_x = self._model_dataframe(data,formula=self.formula_x)
        
        # 点预测
        pred = self.mod.predict(new_x).flatten()
        # 区间预测
        if (ci_method is not None) and (self.coef_dist is not None):
            interval = self._predict_interval(new_x,ci_level=ci_level,method=ci_method)
        else:
            interval = None
        
        predictions = [pd.DataFrame({'mean' : pred}),interval]
        if new_data is None and data_grid is None:
            predictions.append(self.data)
        else:
            predictions.append(data)
        predictions = pd.concat(predictions,axis=1)
        
        return predictions
    
    def _predict(self, new_data=None, new_x=None) -> np.ndarray:
        self._check_fitted()
        
        if new_data is None and new_x is None:
            raise Exception('WRONG')
        
        if new_data is not None:
            new_x     = self._model_dataframe(new_data,formula=self.formula_x)
        pred = self.mod.predict(new_x).flatten()
        
        return pred
    
    def _predict_interval(self,new_x,method,ci_level) -> pd.DataFrame:
        alpha = 1-ci_level
        if method == 'conformal':
            from mapie.regression import MapieRegressor
            pred_interval = MapieRegressor(self.mod).fit(self.x,self.y).predict(new_x,alpha=alpha)[1]
            interval = pd.DataFrame({
                'obs_ci_low':pred_interval[:,0,:].flatten(),
                'obs_ci_up':pred_interval[:,1,:].flatten()
            })
        
        elif method == 'bootstrap':
            pred_dist = self.bootstrap_pred(new_x)
            mean_se   = np.std(pred_dist,axis=0)
            mean_ci_lower ,mean_ci_upper  = np.quantile(pred_dist,[alpha/2,1-alpha/2],axis=0)
            resid_low, resid_up = np.quantile(self.fit_resid,[alpha/2,1-alpha/2])
            obs_ci_lower = resid_low + mean_ci_lower
            obs_ci_upper  = resid_up + mean_ci_upper
            interval = pd.DataFrame({
                'mean_se'      : mean_se,
                'mean_ci_lower': mean_ci_lower,
                'mean_ci_upper': mean_ci_upper,
                'obs_ci_lower' : obs_ci_lower,
                'obs_ci_upper' : obs_ci_upper,
            })
            
        else:
            raise Exception('WRONG Method')
        
        return interval
    
    def plot_prediction(
        self,
        data_grid:dict,
        ci_type  :Union[str,list] = 'mean',
        **predict_kwargs
    ):
        
        predict_kwargs.update({'data_grid':data_grid})
        
        prediction = self.prediction(**predict_kwargs)
        plot_var   = list(predict_kwargs['data_grid'].keys())
        plot       = plot_grid(
            data     = prediction,
            plot_var = plot_var,
            ci_type  = ci_type,
            y_label  = self.y_col
        )

        return plot
    
    def plot_all_prediction(
        self,
        plot_x  :list            = None,
        color_by:dict            = None,
        free_y  :bool            = False,
        ci_type :Union[str,list] = 'mean',
    ):
        if plot_x is None:
            plot_x = self.data.drop(self.y_col,axis=1).columns
        else:
            plot_x = pd.Index(plot_x)
        
        color_by = color_by if color_by is not None else {}
        if len(color_by) == 0:
            color_x = None
        elif len(color_by) == 1:
            color_x = list(color_by.keys())[0]
        elif len(color_by) > 1:
            raise Exception('WRONG color_by')
        
        pred_grid = []
        for x in plot_x:
            if x in color_by.keys():
                continue
            data_grid = {x:'line',**color_by}
            pred = (
                self.prediction(data_grid=data_grid)
                .drop(plot_x.difference(data_grid.keys()),axis=1)
                .rename({x:'x_value'},axis=1)
                .assign(x_name=x)
            )
            pred_grid.append(pred)
        pred_grid = pd.concat(pred_grid,axis=0)
        
        plot = plot_all_grid(
            data    = pred_grid,
            free_y  = free_y,
            ci_type = ci_type,
            y_label = self.y_col,
            color_x = color_x
        )
        
        return plot
    
    def compare_prediction(
        self,
        data_grid:dict,
        ci_level    = 0.95,
        hypothesis  = 0,
        alternative = 'two_side',
    ) -> pd.DataFrame:
        data      = DataGrid(self.data.drop(self.y_col,axis=1)).get_grid(**data_grid) # TODO 检查datagrid的合规性
        x         = self._model_dataframe(data,formula=self.formula_x)
        alpha     = 1 - ci_level
        
        def shift_diff(x):
            shift = np.roll(x,shift=1,axis=1)
            diff  = (x - shift)[:,1:]
            return diff
        def contrast_data(data):
            raw_data   = data.iloc[:-1,:].round(2).astype('str')
            shift_data = data.shift(-1).iloc[:-1,:].convert_dtypes().round(2).astype('str')
            result     = raw_data + '->' + shift_data
            change_col = []
            for col in result.columns:
                if col not in self.formula:
                    continue
                if (raw_data.loc[:,col] == shift_data.loc[:,col]).all():
                    continue
                change_col.append(col)
            result     = result.loc[:,change_col]
            return result
        
        # 均值的预测值对比
        pred            = self.bootstrap_pred(new_x=x)
        shift_diff_pred = shift_diff(pred)
        diff            = shift_diff_pred.mean(axis=0)
        diff_mean_std   = shift_diff_pred.std(axis=0)
        diff_mean_p     = []
        for i in range(shift_diff_pred.shape[1]):
            p_value = get_p_value(shift_diff_pred[:,i],hypothesis=hypothesis,alternative=alternative)
            diff_mean_p.append(p_value)
        diff_mean_lower ,diff_mean_upper  = np.quantile(shift_diff_pred,[alpha/2,1-alpha/2],axis=0)
        
        # 单个观测的预测值对比
        rng                 = np.random.default_rng(seed=self.rng_seed)
        pred_obs            = pred + rng.choice(self.fit_resid,size=pred.shape,replace=True)
        shift_diff_pred_obs = shift_diff(pred_obs)
        diff_obs_std        = shift_diff_pred_obs.std(axis=0)
        diff_obs_p          = []
        for i in range(shift_diff_pred_obs.shape[1]):
            p_value = get_p_value(shift_diff_pred_obs[:,i],hypothesis=hypothesis,alternative=alternative)
            diff_obs_p.append(p_value)
        diff_obs_lower ,diff_obs_upper  = np.quantile(shift_diff_pred_obs,[alpha/2,1-alpha/2],axis=0)
        
        comparison = pd.DataFrame({
            'diff'        : diff,
            'mean_std'    : diff_mean_std,
            'mean_p_value': diff_mean_p,
            'mean_lower'  : diff_mean_lower,
            'mean_upper'  : diff_mean_upper,
            'obs_std'     : diff_obs_std,
            'obs_p_value' : diff_obs_p,
            'obs_lower'   : diff_obs_lower,
            'obs_upper'   : diff_obs_upper
        })
        comparison = pd.concat([contrast_data(data),comparison],axis=1)
        return comparison
    
    def _check_fitted(self):
        if self.mod is None:
            raise Exception('Need fit first')
    

def get_conf_int():
    ...

def get_p_value(sample:np.ndarray, hypothesis:float, alternative='two_side') -> float:
    sample    = sample.flatten()
    n         = len(sample)
    hypothesis = abs(hypothesis)

    n_less    = np.sum(sample>hypothesis)
    n_greater = np.sum(sample<hypothesis)
    
    if alternative == 'two_side':
        p_value = min(n_less,n_greater) / n * 2
    elif alternative == 'greater':
        p_value = n_greater / n
    elif alternative == 'less':
        p_value = n_less / n
    else:
        raise Exception(f'WRONG alternative({alternative})')
        
    return p_value

def get_progress_bar(iter,show_progress):
    if show_progress == False:
        return iter
    try:
        from tqdm import tqdm
        iter = tqdm(iter)
    except:
        pass
    return iter