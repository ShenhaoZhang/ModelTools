import warnings

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

class Novelty:
    #TODO 增加可视化分析 ts_scatter
    def __init__(self,train_x:pd.DataFrame,test_x:pd.DataFrame) -> None:
        self.train_x = train_x
        self.test_x = test_x
        
        self.score = {}
        self.score_ci = {}
    
    def get_score(self,method='lof',save=True):
        score = eval(f'self._score_{method}(self.train_x,self.test_x,{save})')
        return score
    
    def get_score_CI(self,method='lof',n_bootstrap=1000,alpha=0.05):
        warnings.filterwarnings(action='ignore',message='.*does not have valid feature names.*')
        score_bootstrap = []
        for i in tqdm(range(n_bootstrap)):
            data = self.train_x.sample(frac=1,replace=True)
            score = eval(f'self._score_{method}(data,self.test_x,False)')
            score_bootstrap.append(score)
        low    = np.quantile(score_bootstrap,q=alpha/2,axis=0)
        median = np.quantile(score_bootstrap,q=0.5,axis=0)
        high   = np.quantile(score_bootstrap,q=1-alpha/2,axis=0)
        self.score_ci[method] = {'low':low,'median':median,'high':high}
        return low,median,high
    
    def _score_lof(self,train_x,test_x,save=True):
        clf = LocalOutlierFactor(novelty=True)
        clf.fit(train_x)
        score = clf.score_samples(test_x)
        if save is True:
            self.score['lof'] = score
        return score
    
    def _score_gmm(self,train_x,test_x,save=True):
        #TODO 单独拿出来，放在model中
        from sklearn.mixture import GaussianMixture
        from sklearn.model_selection import GridSearchCV

        cv = GridSearchCV(
            estimator=GaussianMixture(max_iter=2000,random_state=0),
            param_grid={'covariance_type':['full','diag'],'n_components':[1,2]}
        )
        cv.fit(train_x)
        return cv.best_estimator_.score_samples(test_x)
    
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import plotnine as gg
    from sklearn.datasets import make_spd_matrix
    
    n = 10
    p = 1000
    
    rng = np.random.default_rng(0)
    mean = np.zeros(shape=n)
    sigma = make_spd_matrix(n_dim=n,random_state=0)
    x = rng.multivariate_normal(mean=mean,cov=sigma,size=p)
    df = pd.DataFrame(data=x,columns=[f'x{i}' for i in range(n)])
    
    dist = np.sqrt(df.x0**2+df.x1**2+df.x2**2+df.x3**2+df.x4**2)
    df_normal = df.loc[dist<3]
    df_train = df_normal.iloc[:-100,:]
    df_test = pd.concat([df.loc[dist>=3],df_normal.iloc[-100:]],axis=0)
    print(df_train)
    print(df_test)
    nov = Novelty(train_x=df_train,test_x=df_test)
    score = nov.get_score(method='gmm')
    print(score)
    
    # low,median,high = nov.get_score_CI(method='gmm')
    # plot = (
    #     pd.DataFrame({'score':score,'low':low,'high':high})
    #     .reset_index()
    #     .pipe(gg.ggplot)
    #     + gg.aes(x='index',y='score')
    #     + gg.geom_point()
    #     + gg.geom_line(gg.aes(x='index',y='low'))
    #     + gg.geom_line(gg.aes(x='index',y='high'))
    # )
    # plot = (
    #     pd.DataFrame({'score':score,'diff':high-low,
    #                   'dist':np.sqrt(df_test.x0**2+df_test.x1**2+df_test.x2**2+df_test.x3**2+df_test.x4**2)})
    #     .reset_index()
    #     .pipe(gg.ggplot)
    #     + gg.aes(x='score',y='dist')
    #     + gg.geom_point()
    # )
    # print(plot)