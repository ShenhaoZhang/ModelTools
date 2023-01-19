from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble as en
from sklearn import metrics
from sklearn import preprocessing as pr

from ._base import BaseBuilder


class CentralRegBuilder(BaseBuilder):
    
    def __init__(self, cv_method, cv_split, cv_shuffle, cv_score:str = 'mse',param_type:str = 'fast') -> None:
        super().__init__(cv_method, cv_split, cv_shuffle, cv_score, param_type)
        
        self.preprocess = {
            'std'  : pr.StandardScaler(),
            'poly' : pr.PolynomialFeatures(),
            'inter': pr.PolynomialFeatures(interaction_only=True),
            'sp'   : pr.SplineTransformer()
        }
        
        self.param_fast = {
            'poly__degree'         : [2,3,4],
            'inter__degree'        : [2,3],
            'sp__extrapolation'    : ['constant','continue','linear'],
            'sp__knots'            : ['uniform','quantile'],
            'EN__l1_ratio'         : [.1,.5,.7,.9,.95,.99,1],
            'HUBER__alpha'         : [0.0001],
            'HUBER__epsilon'       : [1.35],
            'DT__max_depth'        : [2,4,6,8,10],
            'DT__min_samples_split': [5,30,90,200],
            'RF__max_features'     : [1,'sqrt'],
        }
        self.param_complete = self.param_fast.copy()
        self.param_complete.update({
            'HUBER__alpha'  : [0.0001,0.001,0.01,0.1,1,10,100],
            'HUBER__epsilon': [1,1.05,1.1,1.15,1.2,1.25,1.3,1.35],
        })
        self.param = self.param_fast if self.param_type == 'fast' else self.param_complete
        
        self.model = {
            'OLS'  : lm.LinearRegression(), 
            'LAR'  : lm.Lars(),
            'HUBER': lm.HuberRegressor(max_iter=100),
            'EN'   : lm.ElasticNet(random_state=0),
            'BR'   : lm.BayesianRidge(),
            'DT'   : tree.DecisionTreeRegressor(random_state=0),
            'RF'   : en.RandomForestRegressor(random_state=0),
        }
        
        self.struct = {
            'lm' : [
                'OLS',        'inter_OLS',        'poly_OLS',       'sp_OLS',       'inter_sp_OLS',        
                'std_HUBER',  'inter_std_HUBER',  'poly_std_HUBER', 'sp_std_HUBER', 'inter_sp_std_HUBER',  
                'std_EN',     'inter_std_EN',     'poly_std_EN',    'sp_std_EN',    'inter_sp_std_EN',     
                'std_LAR',    'inter_std_LAR',    'poly_std_LAR',   'sp_std_LAR',   'inter_sp_std_LAR',    
            ],
            'tr' : [
                'DT', 'RF'
            ]
        }
        
    @staticmethod
    def get_cv_score(score_name):
        score = {
            'mse' : metrics.get_scorer('neg_mean_squared_error'),  
            'mae' : metrics.get_scorer('neg_mean_absolute_error'),
            'mdae': metrics.get_scorer('neg_median_absolute_error'),
            'mape': metrics.get_scorer('neg_mean_absolute_percentage_error'),
        }
        score = score.get(score_name)
        return score