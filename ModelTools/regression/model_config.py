from sklearn import linear_model as lm

_linear_model = {
    'OLS'  : lm.LinearRegression,
    'HUBER': lm.HuberRegressor,
    'EN'   : lm.ElasticNetCV,
    'LASSO': lm.Lasso,
    'QR'   : lm.QuantileRegressor,
}


_default_param = {
    'QR' : {
        'solver' : 'highs'
    }
}


_cv_param_grid = {
    'QR' : {
        'alpha' : [0,.1,.5,.7,.9,.95,.99,1]
    },
    'LASSO':{
        'alpha' : [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1e+00, 1e+01,1e+02, 1e+03, 1e+04, 1e+05, 1e+06]
    },
}