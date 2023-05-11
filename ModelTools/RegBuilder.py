from typing import Union

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

from .Metric import Metric

class RegBuilder:
    
    model = {
        'OLS'  : lm.LinearRegression, 
        'LAR'  : lm.Lars,
        'HUBER': lm.HuberRegressor,
        'EN'   : lm.ElasticNet,
        'BR'   : lm.BayesianRidge,
        'DT'   : tree.DecisionTreeRegressor,
        'RF'   : en.RandomForestRegressor,
    }
    preprocess = {
        'std'  : pr.StandardScaler,
        'poly' : pr.PolynomialFeatures,
        'sp'   : pr.SplineTransformer
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
        method:Union[list,dict]
    ) -> None:
        
        # 两种method：
        # list ：['poly_OLS']
        # dict ：{'OLS':{'poly':{'degree':2}}}
        self.method = method
        
        self.estimator = {}
        self.init_estimator()
        
    def init_estimator(self):
        
        if isinstance(self.method,dict):
            ...
        
        elif isinstance(self.method,list):
            ...
    
    def fit(self) -> dict:
        ...
    
    def predict_point(self) -> dict:
        ...
    
    def predict_interval(self) -> dict:
        ...
    