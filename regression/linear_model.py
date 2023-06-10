import re
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn import linear_model as lm
from sklearn.base import clone
from patsy import dmatrices,dmatrix

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
        data:dict|pd.DataFrame
    ) -> None:
        self.formula   = formula
        self.data      = self.__init_data(data) 
        self.x,self.y  = self.__model_dataframe(data=self.data,formula=self.formula)
        self.mod       = None
        self.boot_dist = None
    
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
        return self
    
    def coef(self,CI_level=0.95) -> pd.DataFrame:
        self.__check_fitted()
        
        coef_name  = self.x.columns
        coef_value = self.mod.coef_.flatten()
        coef       = pd.DataFrame(data={'Estimate':coef_value},index=coef_name)
        
        if self.boot_dist is not None:
            low_level  = (1 - CI_level) / 2
            high_level = CI_level + (1 - CI_level) / 2
            coef = (
                pd.concat([
                    coef,
                    pd.DataFrame({
                        'Std_Error': np.std(self.boot_dist,axis=1),
                        'CI_Low'   : np.quantile(self.boot_dist,low_level,axis=1),
                        'CI_High'  : np.quantile(self.boot_dist,high_level,axis=1),
                        },index=coef_name)
                ],axis=1)
            )
            coef['z'] = coef.Estimate / coef.Std_Error
            coef = coef.loc[:,['Estimate','Std_Error','z','CI_Low','CI_High']]
        
        return coef
    
    def predict(self,new_data:pd.DataFrame=None,alpha=0.05,ci_method='conformal') -> pd.DataFrame:
        self.__check_fitted()
        
        if new_data is None:
            data = self.data
        else:
            data = self.__init_data(new_data)
            
        formula_x = re.findall('~(.+)',self.formula)[0]
        new_x     = self.__model_dataframe(data,formula=formula_x)
        
        # 点预测
        pred = self.mod.predict(new_x).flatten()
        # 区间预测
        interval = self.__predict_interval(new_x,alpha=alpha,method=ci_method)
        
        predictions = pd.concat([pd.DataFrame({'pred' : pred}),interval],axis=1)
        
        if new_data is None:
            predictions = pd.concat([predictions,self.data],axis=1)
        
        return predictions

    def bootstrap_param(self,n_resamples=1000) -> np.ndarray:
        self.__check_fitted()
        
        mod            = clone(self.mod)
        bootstrap_x    = self.x.to_numpy()
        bootstrap_data = [bootstrap_x[:,i] for i in range(bootstrap_x.shape[1])]
        bootstrap_data.append(self.y)
        def get_boot_coef(*args):
            x = np.column_stack(args[:-1])
            y = args[-1]
            mod.fit(x,y)
            return mod.coef_
        self.boot_dist = bootstrap(
            data             = bootstrap_data,
            statistic        = get_boot_coef,
            n_resamples      = n_resamples,
            confidence_level = 0.5,
            paired           = True,
            random_state     = 0
        ).bootstrap_distribution
        return self.boot_dist
    
    def __predict_interval(self,new_x,method='conformal',alpha=0.05) -> pd.DataFrame:
        if method == 'conformal':
            from mapie.regression import MapieRegressor
            pred_interval = MapieRegressor(self.mod).fit(self.x,self.y).predict(new_x,alpha=alpha)[1]
            interval = pd.DataFrame({
                'pred_low':pred_interval[:,0,:].flatten(),
                'pred_up':pred_interval[:,1,:].flatten()
            })
        
        elif method == 'bootstrap':
            if self.boot_dist is None:
                self.bootstrap_param()
            param_dist = self.boot_dist.T
            pred_dist = []
            mod = clone(self.mod).fit(self.x,self.y)
            for param in param_dist:
                mod.coef_ = param
                pred = mod.predict(new_x)
                pred_dist.append(pred)
            pred_dist = np.column_stack(pred_dist)
            ci_low ,ci_up  = np.quantile(pred_dist,[alpha/2,1-alpha/2],axis=1)
            resid_low,resid_up = np.quantile(self.train_resid,[alpha/2,1-alpha/2])
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
    
    def __check_fitted(self):
        if self.mod is None:
            raise Exception('Need fit first')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng(0)
    x   = rng.normal(0,0.1,1000).cumsum()
    y   = rng.standard_t(df=1,size=1000)+3+2*x+x**2
    m   = LinearModel('y~x+I(x*x)',data={'x':x,'y':y}).fit(method='HUBER')
    m.bootstrap_param(n_resamples=1000)
    print('coef',m.coef())
    print(m.predict(ci_method='bootstrap'))
    
    # pred = m.predict()
    # print(pred)
    
    # x_new = rng.normal(0,1,10)
    # print(m.predict(new_data={'x':x_new}))
    
    # plt.scatter(x,y)
    # plt.plot(x,m.predict(),c='red')
    # plt.show()