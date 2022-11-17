from sklearn.base import clone
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model as lm
from sklearn import preprocessing as pr
from sklearn.decomposition import PCA 

preprocess = {
    'poly':pr.PolynomialFeatures(),
    'inter':pr.PolynomialFeatures(interaction_only=True),
    'sp':pr.SplineTransformer()
}

model = {
    'ols':lm.LinearRegression(),
    'huber':lm.HuberRegressor()
}

param = {
    'poly__degree':[1,2,3],
    'inter__degree':[1,2,3]
}


    


    

if __name__ == '__main__':
    print(get_pipeline('poly_huber',param=param))