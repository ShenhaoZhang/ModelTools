from sklearn import linear_model as lm
from sklearn import preprocessing as pr

preprocess = {
    'std'  : pr.StandardScaler(),
    'poly' : pr.PolynomialFeatures(),
    'inter': pr.PolynomialFeatures(interaction_only=True),
    'sp'   : pr.SplineTransformer()
}

model = {
    'OLS'  : lm.LinearRegression(),
    'HUBER': lm.HuberRegressor(max_iter=10000),
    'EN'   : lm.ElasticNetCV(),
    'QR'   : lm.QuantileRegressor(solver='highs',quantile=0.5,alpha=0)
}

param = {
    'poly__degree':[1,2,3],
    'inter__degree':[1,2,3]
}

base_struct = [
    'poly_OLS', 'inter_sp_OLS',
    'poly_std_HUBER', 'inter_sp_std_HUBER',
    'poly_std_EN', 'inter_sp_std_EN',
    'poly_std_QR', 'inter_sp_std_QR'
]