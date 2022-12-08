from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import(
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn import metrics
from sklearn import preprocessing as pr

from ._base import BaseBuilder

class QuantileRegBuilder(BaseBuilder):
    def __init__(
        self, 
        cv_method, 
        cv_split, 
        cv_shuffle, 
        cv_score:str,
        quantile:float
    ) -> None:
        super().__init__(cv_method, cv_split, cv_shuffle, cv_score)
        self.quantile = quantile
        
        self.preprocess = {
            'std'  : pr.StandardScaler(),
            'poly' : pr.PolynomialFeatures(),
            'inter': pr.PolynomialFeatures(interaction_only=True),
            'sp'   : pr.SplineTransformer()
        }
        self.param = {
            'poly__degree'         : [2,3,4],
            'inter__degree'        : [1,2,3],
            'sp__extrapolation'    : ['constant','continue','linear'],
            'sp__knots'            : ['uniform','quantile'],
            'QR__alpha'            : [0,.1,.5,.7,.9,.95,.99,1],
            'GB__learning_rate'    : [.05, .1, .2],
            'GB__max_depth'        : [2, 5, 10],
            'GB__min_samples_leaf' : [1, 5, 10, 20],
            'GB__min_samples_split': [5, 10, 20, 30, 50]
        }

        self.model = {
            'QR' : QuantileRegressor(solver='highs',quantile=self.quantile),
            'GB' : GradientBoostingRegressor(loss='quantile',alpha=self.quantile),
            'HGB': HistGradientBoostingRegressor(loss='quantile',quantile=self.quantile)
        }
        
        self.struct = {
            'lm': ['QR', 'poly_QR', 'sp_QR', 'inter_sp_QR'],
            'tr': ['GB', 'HGB']
        }
        
        self.score = self.get_cv_score(self.cv_score,self.quantile)
        
    @staticmethod
    def get_cv_score(score_name,quantile=None):
        score = {
            'pinball': metrics.make_scorer(
                metrics.mean_pinball_loss,
                alpha = quantile,
                greater_is_better=False
            )
        }
        score = score.get(score_name)
        return score