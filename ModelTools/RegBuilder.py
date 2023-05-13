from typing import Union

import numpy as np
from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble as en
from sklearn import preprocessing as pr
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
        'std'  : pr.StandardScaler(),
        'poly' : pr.PolynomialFeatures(),
        'inter': pr.PolynomialFeatures(interaction_only=True),
        'sp'   : pr.SplineTransformer()
    }
    param = {
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
    }
    
    def __init__(
        self,
        method:Union[str,dict] = 'poly_OLS'
    ) -> None:
        
        # 两种method：
        # list ：poly_OLS
        # dict ：{'poly':{'degree':2},'OLS':{}}
        
        self.method    = method
        self.estimator = None
        
        self.init_method()
        self.init_pipe()
    
    def init_method(self) -> None:
        if isinstance(self.method,str):
            method_split = self.method.split('_')
            method = {}
            for mth in method_split:
                param       = {name:param for name,param in self.param.items() if name.split('__')[0]==mth}
                method[mth] = param
        
        if isinstance(self.method,dict):
            method = {}
            for mth,param in self.method.items():
                adj_param = {f'{mth}__{n}':v for n,v in param.items()}
                method[mth] = adj_param
        
        self.method = method
        
    def init_pipe(self):
        pipe = []
        for mth in self.method.keys():
            step = self.preprocess.get(mth,self.model.get(mth))
            pipe.append((mth,step))
        self.pipe = Pipeline(steps=pipe)
    
    def fit(
        self,
        X,y,
        cv_method   = 'kfold',
        cv_n_splits = 5,
        cv_shuffle  = False
    ) -> GridSearchCV:
        
        param_grid = {}
        for param in self.method.values():
            param_grid.update(param)
        
        if cv_method == 'kfold':
            cv_method = KFold(
                n_splits     = cv_n_splits,
                shuffle      = cv_shuffle,
                random_state = 0 if cv_shuffle == True else None
            )
        elif cv_method == 'ts':
            cv_method = TimeSeriesSplit(
                n_splits = cv_n_splits
            )
        
        self.cv = GridSearchCV(
            estimator          = self.pipe,
            param_grid         = param_grid,
            # scoring            = ...,
            # n_jobs             = ...,
            refit              = True,
            cv                 = cv_method,
            # verbose            = ...,
            # pre_dispatch       = ...,
            # error_score        = ...,
            return_train_score = True
        )
        self.cv.fit(
            X,y
        )
        self.X = X 
        self.y = y
        
        return self.cv
    
    def predict(self,X,) -> np.ndarray:
        pred = self.cv.predict(X)
        return pred
    
    def predict_interval(self,X,alpha=0.05) -> tuple:
        #TODO 保存模型，避免重复训练
        from mapie.regression import MapieRegressor
        pred_interval = MapieRegressor(self.cv).fit(self.X,self.y).predict(X,alpha=alpha)[1]
        pred_low      = pred_interval[:,0,:].flatten()
        pred_up       = pred_interval[:,1,:].flatten()
        return pred_low,pred_up

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