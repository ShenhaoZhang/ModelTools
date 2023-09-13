from sklearn import linear_model as lm

_linear_model = {
    'OLS'  : lm.LinearRegression,
    'HUBER': lm.HuberRegressor,
    'EN'   : lm.ElasticNetCV,
    'LASSO': lm.Lasso,
    'QR'   : lm.QuantileRegressor,
    'GM'   : lm.GammaRegressor,
    'TR'   : lm.TweedieRegressor
}


_default_param = {
    'QR' : {
        'solver': 'highs',
        'alpha' : 0
    },
    'GM' : {
        'max_iter': 1000,
        'alpha'   : 0
    },
    'TR' : {
        'max_iter': 1000,
        'alpha'   : 0
    }
}


_cv_param_grid = {
    # 'QR' : {
    #     'alpha' : [0,.1,.5,.7,.9,.95,.99,1]
    # },
    'LASSO':{
        'alpha' : [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1e+00, 1e+01,1e+02, 1e+03, 1e+04, 1e+05, 1e+06]
    },
    'TR' : {
        'power':[0] + list(range(1,3,10))
    }
}