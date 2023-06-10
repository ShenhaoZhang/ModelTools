import re
from typing import Union

import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from formulaic import model_matrix

class RegBuilder:
    def __init__(
        self,
        formula:str,
        data:dict|pd.DataFrame
    ) -> None:
        self.formula  = formula
        self.data     = self.__init_data(data) 
        self.x,self.y = self.__init_x_and_y(data=data,formula=formula)
    
    def __init_data(self,data) -> pd.DataFrame:
        if isinstance(data,pd.DataFrame):
            return data
        if isinstance(data,dict):
            return pd.DataFrame(data)
    
    def __init_x_and_y(self,data,formula) -> Union[tuple,pd.DataFrame]:
        model_data = model_matrix(data,self.formula)
        if '~' in formula:
            x = model_data.rhs
            y = model_data.lhs.values.flatten()
            return x,y
        else:
            x = model_data
            return x
        
    def __get_coef(self):
        coef_name = self.x.columns
        coef_value = self.mod.coef_.flatten()
        coef = pd.Series(data=coef_value,index=coef_name,name='coef')
        return coef
    
    def fit(self,method='OLS',method_kwargs:dict=None):
        model = {
            'OLS'  : lm.LinearRegression,
            'HUBER': lm.HuberRegressor
        }
        method_kwargs = {} if method_kwargs is None else method_kwargs
        self.mod = model[method](fit_intercept=False,**method_kwargs)
        self.mod.fit(X=self.x,y=self.y)
        self.coef = self.__get_coef()
        return self
    
    def predict(self,new_data:pd.DataFrame=None,alpha=0.05):
        if new_data is None:
            new_data = self.data
        data      = self.__init_data(new_data)
        formula_x = re.findall('~(.+)',self.formula)[0]
        x         = self.__init_x_and_y(data,formula=formula_x)
        
        from mapie.regression import MapieRegressor
        pred = self.mod.predict(new_data).flatten()
        pred_interval = MapieRegressor(self.mod).fit(self.x,self.y).predict(x,alpha=alpha)[1]
        pred_low      = pred_interval[:,0,:].flatten()
        pred_up       = pred_interval[:,1,:].flatten()
        predictions = pd.DataFrame({
            'pred'    : pred,
            'pred_low': pred_low,
            'pred_up' : pred_up
        })
        
        return predictions

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng(0)
    x   = rng.normal(0,0.1,1000).cumsum()
    y   = rng.standard_t(df=1,size=1000)+3+2*x+x**2
    m   = RegBuilder('y~poly(x,2,raw=True)',data={'x':x,'y':y}).fit(method='HUBER')
    print('coef',m.coef)
    
    # pred = m.predict()
    # print(pred)
    
    # plt.scatter(x,y)
    # plt.plot(x,m.predict(),c='red')
    # plt.show()