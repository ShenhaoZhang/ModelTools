from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble as en
from sklearn import metrics
from sklearn import preprocessing as pr

from ._base import BaseBuilder


class MeanRegBuilder(BaseBuilder):
    
    def __init__(self, cv_method, cv_split, cv_shuffle, cv_score:str = 'mse') -> None:
        super().__init__(cv_method, cv_split, cv_shuffle, cv_score)
        
        self.preprocess = {
            'std'  : pr.StandardScaler(),
            'poly' : pr.PolynomialFeatures(),
            'inter': pr.PolynomialFeatures(interaction_only=True),
            'sp'   : pr.SplineTransformer()
        }
        self.param = {
            'poly__degree'         : [1,2,3],
            'inter__degree'        : [1,2,3],
            'sp__extrapolation'    : ['constant','continue','linear'],
            'sp__knots'            : ['uniform','quantile'],
            'DT__max_depth'        : [2,4,6,8,10],
            'DT__min_samples_split': [5,30,90,200],
            'RF__max_features'     : [1,'sqrt'],
        }
        
        self.model = {
            'OLS'  : lm.LinearRegression(), 
            'LAR'  : lm.Lars(normalize=False),
            'HUBER': lm.HuberRegressor(),
            'EN'   : lm.ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,.99,1],random_state=0),
            # 'QR'   : lm.QuantileRegressor(solver='highs',quantile=0.5,alpha=0),  #TODO 条件alpha
            'BR'   : lm.BayesianRidge(),
            'DT'   : tree.DecisionTreeRegressor(random_state=0),
            'RF'   : en.RandomForestRegressor(random_state=0),
        }
        
        self.struct = {
            'lm' : [
                'OLS',        'poly_OLS',       'sp_OLS',       'inter_sp_OLS',        
                'std_HUBER',  'poly_std_HUBER', 'sp_std_HUBER', 'inter_sp_std_HUBER',  
                'std_EN',     'poly_std_EN',    'sp_std_EN',    'inter_sp_std_EN',     
                'std_LAR',    'poly_std_LAR',   'sp_std_LAR',   'inter_sp_std_LAR',    
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