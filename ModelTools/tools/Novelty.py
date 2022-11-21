import warnings

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

class Novelty:
    def __init__(self,train_x:pd.DataFrame,test_x:pd.DataFrame) -> None:
        self.train_x = train_x
        self.test_x = test_x
    
    def get_score(self,method='lof'):
        score = eval(f'self.score_{method}(self.train_x,self.test_x)')
        return score
    
    def get_score_CI(self,method='lof',n_bootstrap=1000):
        warnings.filterwarnings(action='ignore',message='.*does not have valid feature names.*')
        score_bootstrap = []
        for i in tqdm(range(n_bootstrap)):
            data = self.train_x.sample(frac=1,replace=True)
            score = self.score_lof(data,self.test_x)
            score_bootstrap.append(score)
        low = np.quantile(score_bootstrap,q=0.025,axis=0)
        high = np.quantile(score_bootstrap,q=0.975,axis=0)
        return low,high
    
    def score_lof(self,train_x,test_x):
        clf = LocalOutlierFactor(novelty=True)
        clf.fit(train_x)
        score = clf.score_samples(test_x)
        return score
    
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import plotnine as gg
    from sklearn.datasets import make_spd_matrix
    
    n = 5
    p = 1000
    
    rng = np.random.default_rng(0)
    mean = np.zeros(shape=5)
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
    score = nov.get_score(method='lof')
    
    low,high = nov.get_score_CI()
    plot = (
        pd.DataFrame({'score':score,'low':low,'high':high})
        .reset_index()
        .pipe(gg.ggplot)
        + gg.aes(x='index',y='score')
        + gg.geom_point()
        + gg.geom_line(gg.aes(x='index',y='low'))
        + gg.geom_line(gg.aes(x='index',y='high'))
    )
    print(plot)