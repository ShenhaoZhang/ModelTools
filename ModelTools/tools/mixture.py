from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Mixture:
    def __init__(self,data,col_x) -> None:
        self.data = data
        self.col_x = col_x
        
        self.cv = None
        self.param = None
    
    def fit_gmm(self,pca:float=0.95,pca_scale:bool=True):
        param_grid = {
            'gmm__covariance_type':['full', 'tied', 'diag', 'spherical'],
            'gmm__n_components':list(range(1,11)),
        }
        steps = [('gmm',GaussianMixture(max_iter=1000,random_state=0))]
        
        if pca is not None:
            step_pca = [
                ('scale',StandardScaler(with_mean=pca_scale,with_std=pca_scale)),
                ('pca',PCA(random_state=0))
            ]
            steps = step_pca + steps
            param_grid['pca__n_components'] = [pca]
            
        self.cv = GridSearchCV(
            estimator  = Pipeline(steps=steps),
            param_grid = param_grid,
            refit      = True,
            n_jobs     = -1
        )
        self.cv.fit(self.data.loc[:,self.col_x])
        self.param = self.cv.best_params_
    
    def predict_llh(self,new_data=None):
        data = new_data if new_data is not None else self.data
        data = data.loc[:,self.col_x]
        llh = self.cv.score_samples(data)
        return llh 