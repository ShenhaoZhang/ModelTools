from typing import Union

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import(
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from .RegBuilder import RegBuilder

class QRegBuilder(RegBuilder):
    
    model = {
        'QR'  : QuantileRegressor(solver='highs'),
        'GBR' : GradientBoostingRegressor(loss='quantile'),
        'HGBR': HistGradientBoostingRegressor(loss='quantile'),
    }
    
    def __init__(
        self, 
        method: str | dict | Pipeline = 'poly_OLS',
        method_args = None,
        formula = None,
        alpha: float = 0.5
    ) -> None:
        self.cv_param.update(
            {
                'QR__alpha'            : [0,.1,.5,.7,.9,.95,.99,1],
                'GB__learning_rate'    : [.05, .1, .2],
                'GB__max_depth'        : [2, 5, 10],
                'GB__min_samples_leaf' : [1, 5, 10, 20],
                'GB__min_samples_split': [5, 10, 20, 30, 50]
            }
        )
        self.alpha = alpha
        super().__init__(method,method_args,formula)
        
    def fit(
        self,
        X,y,
        alpha       = None,
        cv_method   = 'kfold',
        cv_n_splits = 5,
        cv_shuffle  = False,
        cv_score    = 'pinball',
    ):
        if alpha is not None:
            self.alpha         = alpha
            self.model['QR']   = self.model['QR'].set_params(quantile = alpha)
            self.model['GBR']  = self.model['GBR'].set_params(alpha = alpha)
            self.model['HGBR'] = self.model['HGBR'].set_params(quantile = alpha)
            
        return super().fit(X,y,cv_method,cv_n_splits,cv_shuffle,cv_score)

    def predict_interval(self,X) -> tuple:
        ...
    
    
def get_cv_score(score_name:str,alpha:float):
    score = {
        'pinball': metrics.make_scorer(metrics.mean_pinball_loss,alpha = alpha,greater_is_better=False)
    }
    score = score.get(score_name)
    return score
