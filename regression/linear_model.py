import re
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn import linear_model as lm
from sklearn.base import clone
from patsy import dmatrices,dmatrix
import matplotlib.pyplot as plt

class LinearModel:
    linear_model = {
        'OLS'  : lm.LinearRegression,
        'HUBER': lm.HuberRegressor,
        'EN'   : lm.ElasticNetCV,
        'LASSO': lm.LassoCV
    }
    def __init__(
        self,
        formula:str,
        data:Union[dict,pd.DataFrame]
    ) -> None:
        self.formula   = formula
        self.data      = self.__init_data(data) 
        self.x,self.y  = self.__model_dataframe(data=self.data,formula=self.formula)
        self.mod       = None
        self.coef_dist_boot = None
    
    def __init_data(self,data) -> pd.DataFrame:
        if isinstance(data,pd.DataFrame):
            return data
        if isinstance(data,dict):
            return pd.DataFrame(data)
        if isinstance(data,(list,tuple)):
            return pd.DataFrame(data)
    
    def __model_dataframe(self,data,formula) -> Union[tuple,pd.DataFrame]:
        
        def matrix_to_df(matrix):
            return pd.DataFrame(data=np.asarray(matrix),columns=matrix.design_info.column_names)
        
        if '~' in formula:
            matrices = dmatrices(formula,data)
            y = np.asarray(matrices[0]).flatten()
            x = matrix_to_df(matrices[1])
            return x,y
        else:
            matrix = dmatrix(formula,data)
            x = matrix_to_df(matrix) 
            return x
    
    def fit(self,method='OLS',method_kwargs:dict=None):
        method_kwargs = {} if method_kwargs is None else method_kwargs
        self.mod = self.linear_model[method](fit_intercept=False,**method_kwargs)
        self.mod.fit(X=self.x,y=self.y)
        self.train_resid = self.y - self.mod.predict(self.x)
        self.coef_dist_boot = None
        return self

    def check_model(self):
        ...

    def bootstrap_coef(self,n_resamples=1000):
        self.__check_fitted()
        
        #TODO 将模型中的超参数固定（包含CV类型的MOD）
        mod            = clone(self.mod)
        bootstrap_x    = self.x.to_numpy()
        bootstrap_data = [bootstrap_x[:,i] for i in range(bootstrap_x.shape[1])]
        bootstrap_data.append(self.y)
        def get_boot_coef(*args):
            x = np.column_stack(args[:-1])
            y = args[-1]
            mod.fit(x,y)
            return mod.coef_
        self.coef_dist_boot = bootstrap(
            data             = bootstrap_data,
            statistic        = get_boot_coef,
            n_resamples      = n_resamples,
            confidence_level = 0.5,
            paired           = True,
            random_state     = 0,
            vectorized       = False
        ).bootstrap_distribution.T
        
        self.coef_dist_boot = pd.DataFrame(
            data    = self.coef_dist_boot,
            columns = self.x.columns
        )
        return self
    
    def bootstrap_pred(self,new_x) -> np.ndarray:
        if self.coef_dist_boot is None:
            self.bootstrap_coef()
        mod = clone(self.mod).fit(self.x,self.y)
        pred_dist = []
        for coef in self.coef_dist_boot.to_numpy():
            mod.coef_ = coef
            pred = mod.predict(new_x)
            pred_dist.append(pred)
        pred_dist = np.column_stack(pred_dist)
        return pred_dist
    
    def get_coef(self, hypothesis:float=0, alternative='two_side', CI_level=0.95) -> pd.DataFrame:
        self.__check_fitted()
        
        coef_name  = self.x.columns
        coef_value = self.mod.coef_.flatten()
        coef       = pd.DataFrame(data={'Estimate':coef_value},index=coef_name)
        
        # 用bootstrap方法对参数进行统计推断
        if self.coef_dist_boot is not None:
            low_level  = (1 - CI_level) / 2
            high_level = CI_level + (1 - CI_level) / 2
            ci_low     = np.quantile(self.coef_dist_boot,low_level,axis=0)
            ci_high    = np.quantile(self.coef_dist_boot,high_level,axis=0)
            std_error  = np.std(self.coef_dist_boot,axis=0)
            z_score    = (coef.Estimate - hypothesis) / std_error
            
            p_value = []
            for name in coef_name:
                value = get_p_value(
                    self.coef_dist_boot.loc[:,name].to_numpy(),
                    hypothesis  = hypothesis,
                    alternative = alternative,
                )
                p_value.append(value)
            
            summary = pd.DataFrame(
                {
                    'Std_Error': std_error,
                    'Z'        : z_score,
                    'P_value'  : p_value,
                    'CI_Low'   : ci_low,
                    'CI_High'  : ci_high
                },
                index = coef_name
            )
            
            coef = pd.concat([coef,summary],axis=1).loc[:,['Estimate','Std_Error','Z','P_value','CI_Low','CI_High']]
            
        return coef
    
    def get_metric(self,bootstrap=False,summary=True,CI_level=0.95) -> pd.DataFrame:
        from .metric import Metric
        
        metric = Metric(y_true=self.y,y_pred=self.mod.predict(self.x)).get_metric()
        if bootstrap == True:
            bootstrap_pred = list(self.bootstrap_pred(self.x).T)
            metric_boot    = Metric(self.y,bootstrap_pred).get_metric()
            if summary == True:
                low_level    = (1 - CI_level) / 2
                high_level   = CI_level + (1 - CI_level) / 2
                metric_std   = metric_boot.std(axis=0).to_frame().T
                metric_ci    = metric_boot.quantile([low_level,high_level],axis=0)
                metric       = pd.concat([metric,metric_std,metric_ci],axis=0)
                metric.index = ['Estimate','Std_Error','CI_Low','CI_High']
            else:
                metric = metric_boot
        return metric
    
    def plot_coef_dist(self):
        ...
    
    def plot_coef_pair(self):
        ...
    
    def predict(self,new_data:pd.DataFrame=None,alpha=0.05,ci_method='conformal') -> pd.DataFrame:
        self.__check_fitted()
        
        if new_data is None:
            data = self.data
        elif isinstance(new_data,dict):
            from .data_grid import DataGrid
            y_col = re.findall('(.+)~',self.formula)
            data = DataGrid(self.data.drop(y_col,axis=1)).get_grid(**new_data)
        else:
            data = self.__init_data(new_data)
            
        formula_x = re.findall('~(.+)',self.formula)[0]
        new_x     = self.__model_dataframe(data,formula=formula_x)
        
        # 点预测
        pred = self.mod.predict(new_x).flatten()
        # 区间预测
        interval = self.__predict_interval(new_x,alpha=alpha,method=ci_method)
        
        predictions = [pd.DataFrame({'pred' : pred}),interval]
        if new_data is None:
            predictions.append(self.data)
        else:
            predictions.append(data)
        predictions = pd.concat(predictions,axis=1)
        
        return predictions
    
    def __predict_interval(self,new_x,method='conformal',alpha=0.05) -> pd.DataFrame:
        if method == 'conformal':
            from mapie.regression import MapieRegressor
            pred_interval = MapieRegressor(self.mod).fit(self.x,self.y).predict(new_x,alpha=alpha)[1]
            interval = pd.DataFrame({
                'pred_low':pred_interval[:,0,:].flatten(),
                'pred_up':pred_interval[:,1,:].flatten()
            })
        
        elif method == 'bootstrap':
            pred_dist = self.bootstrap_pred(new_x)
            ci_low ,ci_up  = np.quantile(pred_dist,[alpha/2,1-alpha/2],axis=1)
            resid_low, resid_up = np.quantile(self.train_resid,[alpha/2,1-alpha/2])
            pred_low = resid_low + ci_low
            pred_up  = resid_up + ci_up
            interval = pd.DataFrame({
                'pred_low': pred_low,
                'pred_up' : pred_up,
                'ci_low'  : ci_low,
                'ci_up'   : ci_up
            })
            
        else:
            raise Exception('WRONG Method')
        
        return interval
    
    def plot_prediction(self):
        ...
    
    def summary(self):
        # coef metric check
        ...
    
    def __check_fitted(self):
        if self.mod is None:
            raise Exception('Need fit first')

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from statsmodels.formula.api import ols
    
    rng = np.random.default_rng(0)
    x   = rng.normal(0,0.1,1000).cumsum()
    y   = rng.standard_t(df=1,size=1000)+3+2*x+x**2
    
    m   = LinearModel('y~x+I(x**2)+I(x**3)',data={'x':x,'y':y}).fit(method='HUBER')
    m.bootstrap_coef(n_resamples=1000)
    
    print('coef',m.get_coef())
    print(m.predict(ci_method='bootstrap'))
    
    # pred = m.predict()
    # print(pred)
    
    # x_new = rng.normal(0,1,10)
    # print(m.predict(new_data={'x':x_new}))
    
    # plt.scatter(x,y)
    # plt.plot(x,m.predict(),c='red')
    # plt.show()