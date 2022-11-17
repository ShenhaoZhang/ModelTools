from sklearn import linear_model as lm
from sklearn import preprocessing as pr

preprocess = {
    'std':pr.StandardScaler(),
    'poly':pr.PolynomialFeatures(),
    'inter':pr.PolynomialFeatures(interaction_only=True),
    'sp':pr.SplineTransformer()
}

model = {
    'ols':lm.LinearRegression(),
    'huber':lm.HuberRegressor(max_iter=10000)
}

param = {
    'poly__degree':[1,2,3],
    'inter__degree':[1,2,3]
}
