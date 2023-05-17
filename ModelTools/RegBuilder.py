from typing import Union

import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble as en
from sklearn import preprocessing as pr
from sklearn import metrics
from sklearn import decomposition as de
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    KFold
)


class RegBuilder:
    
    model = {
        'OLS'  : lm.LinearRegression(), 
        'LAR'  : lm.Lars(),
        'HUBER': lm.HuberRegressor(max_iter=100),
        'EN'   : lm.ElasticNet(random_state=0),
        'BR'   : lm.BayesianRidge(),
        'DT'   : tree.DecisionTreeRegressor(random_state=0),
        'RF'   : en.RandomForestRegressor(random_state=0),
    }
    preprocess = {
        'std'    : pr.StandardScaler(),
        'poly'   : pr.PolynomialFeatures(),
        'inter'  : pr.PolynomialFeatures(interaction_only=True),
        'sp'     : pr.SplineTransformer(),
        'pca'    : de.PCA(),
        'fml'    : pr.FunctionTransformer()
    }
    cv_param = {
        'poly__degree'         : [2,3,4],
        'inter__degree'        : [2,3],
        'sp__extrapolation'    : ['constant','continue','linear'],
        'sp__knots'            : ['uniform','quantile'],
        'sp__degree'           : [1,2,3],
        'EN__l1_ratio'         : [.1,.5,.7,.9,.95,.99,1],
        'HUBER__alpha'         : [0.0001],
        'HUBER__epsilon'       : [1.35],
        'DT__max_depth'        : [2,4,6,8,10],
        'DT__min_samples_split': [5,30,90,200],
        'RF__max_features'     : [1,'sqrt'],
        'pca__n_components'    : [.9,.95,.99,.999,1]
    }
    
    def __init__(
        self,
        method : Union[str,dict,Pipeline] = 'poly_OLS',
        method_args : dict = None,
        formula:str = None
    ) -> None:
        
        self.method      = method
        self.method_args = method_args
        self.formula     = formula
        self.init_pipeline()
    
        # method type
        # str1  ：poly_OLS
        # str2  : fml_OLS
        # pipe
    
    def init_pipeline(self):
        if isinstance(self.method,Pipeline):
            self.pipeline = Pipeline
            return None
        
        steps = []
        for pipe_name in self.method.split('_'):
            estimator = self.preprocess.get(pipe_name,self.model.get(pipe_name))
            if estimator is None:
                raise Exception(f'未发现{pipe_name}')
            steps.append((pipe_name,estimator))
        self.pipeline = Pipeline(steps)
    
    def update_pipeline(self,estimator_param:dict):
        # 更新pipeline中的参数
        # estimator_param : {'poly__degree':[1]}
        self.pipeline.set_params(**estimator_param)
    
    def get_cv_param_grid(self):
        cv_param_grid = {}
        for pipe_name in self.method.split('_'):
            for param_name,param_value in self.cv_param.items():
                if pipe_name in param_name.split('__')[0]:
                    cv_param_grid.update({param_name:param_value})
        return cv_param_grid
    
    def fit(
        self,
        X : np.ndarray | pd.DataFrame,
        y : np.ndarray | pd.DataFrame,
        cv_method   = 'kfold',
        cv_n_splits = 5,
        cv_shuffle  = False,
        cv_score    = 'mse'
    ):
        if isinstance(X,np.ndarray):
            col_name = [f'x{i}' for i in range(X.shape[1])]
            self.X = pd.DataFrame(data=X,columns=col_name)
        elif isinstance(X,pd.DataFrame):
            self.X = X
        self.y = y
        
        if self.formula is not None:
            mod = self.method.split('_')[-1]
            self.update_pipeline({
                'fml__func'             : formula_transform,
                'fml__kw_args'          : {'formula':self.formula},
                'fml__feature_names_out': lambda x,y: fml_feature_names_out(x,y,formula=self.formula),
            })
            try:
                self.update_pipeline({f'{mod}__fit_intercept':False})
            except:
                pass
        
        cv_param    = self.get_cv_param_grid()
        cv_n_splits = 2 if len(cv_param) == 0 else cv_n_splits
        cv_method   = get_cv_method(cv_method,cv_n_splits,cv_shuffle)
        cv_score    = get_cv_score(cv_score)
        self.cv = GridSearchCV(
            estimator  = self.pipeline,
            param_grid = cv_param,
            scoring    = cv_score,
            n_jobs     = -1,
            refit      = True,
            cv         = cv_method,
        )
        
        self.cv.fit(self.X,self.y)
        self.coef = get_coef(self.cv,fit_intercept=False) if self.formula is not None else get_coef(self.cv)
        
        return self
        
    def predict(self,X=None) -> np.ndarray:
        if X is None:
            X = self.X
        pred = self.cv.predict(X)
        return pred
    
    def predict_interval(self,X=None,alpha=0.05) -> tuple:
        if X is None:
            X = self.X 
        #TODO 保存模型，避免重复训练
        #TODO 选择最好的区间预测方法
        from mapie.regression import MapieRegressor
        pred_interval = MapieRegressor(self.cv).fit(self.X,self.y).predict(X,alpha=alpha)[1]
        pred_low      = pred_interval[:,0,:].flatten()
        pred_up       = pred_interval[:,1,:].flatten()
        return pred_low,pred_up

    def init_data(self,X):
        if isinstance(X,np.ndarray):
            col_name = [f'x{i}' for i in range(X.shape[1])]
            self.X = pd.DataFrame(data=X,columns=col_name)
        elif isinstance(X,pd.DataFrame):
            self.X = X
        self.y = y
        return self.X,self.Y


def get_coef(cv,fit_intercept=True) -> dict:
    coef_value = getattr(cv.best_estimator_[-1],'coef_',[])
    coef_name  = cv.best_estimator_[:-1].get_feature_names_out()
    coef_name  = [f'x{i}' for i in range(len(coef_value))] if coef_name is None else coef_name
    coef       = dict(zip(coef_name,coef_value))
    intercept  = {'intercept_':getattr(cv.best_estimator_[-1],'intercept_',[])}
    coef       = {**intercept,**coef} if fit_intercept == True else {**coef}
    return coef

def get_cv_method(method_name,n_splits,cv_shuffle=False):
    if method_name == 'kfold':
        cv_method = KFold(
            n_splits     = n_splits ,
            shuffle      = cv_shuffle ,
            random_state = 0 if cv_shuffle == True else None
        )
    elif method_name == 'ts':
        cv_method = TimeSeriesSplit(
            n_splits=n_splits
        )
    return cv_method

def get_cv_score(score_name:str):
    score = {
        'mse' : metrics.get_scorer('neg_mean_squared_error'),  
        'mae' : metrics.get_scorer('neg_mean_absolute_error'),
        'mdae': metrics.get_scorer('neg_median_absolute_error'),
        'mape': metrics.get_scorer('neg_mean_absolute_percentage_error'),
    }
    score = score.get(score_name)
    return score

def formula_transform(data:pd.DataFrame,formula:str):
    from formulaic import model_matrix
    return model_matrix(formula,data)   

def fml_feature_names_out(transformer,input_name,formula):
    data = pd.DataFrame(columns=input_name,data=np.ones(len(input_name)).reshape(1,-1))
    output = formula_transform(data,formula)
    return output.columns.to_list()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    rng = np.random.default_rng(0)
    x   = rng.normal(0,0.1,100).cumsum()
    y   = rng.normal(3+2*x+x**2,scale=0.3)
    m   = RegBuilder('poly_OLS')
    m.fit(x.reshape(-1,1),y)

    x_new = np.linspace(-3,3,100)
    y_new = rng.normal(3+2*x_new+x_new**2,scale=0.3)
    mean  = m.predict(x_new.reshape(-1,1))
    lw,up = m.predict_interval(x_new.reshape(-1,1),alpha=0.05)

    plt.plot(x_new,mean,c='red')
    plt.scatter(x_new,y_new)
    plt.fill_between(x_new,lw,up,alpha=0.3)
    plt.vlines(x=min(x),ymin=min(lw),ymax=max(up),colors='green')
    plt.vlines(x=max(x),ymin=min(lw),ymax=max(up),colors='green')
    plt.show()
