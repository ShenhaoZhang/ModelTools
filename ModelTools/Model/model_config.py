from sklearn.base import clone
from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble as en
from sklearn import preprocessing as pr
from sklearn.pipeline import Pipeline 

preprocess = {
    'std'  : pr.StandardScaler(),
    'poly' : pr.PolynomialFeatures(),
    'inter': pr.PolynomialFeatures(interaction_only=True),
    'sp'   : pr.SplineTransformer()
}

model = {
    'OLS'  : lm.LinearRegression(),
    'LAR'  : lm.Lars(normalize=False),
    'HUBER': lm.HuberRegressor(),
    'EN'   : lm.ElasticNetCV(),
    'QR'   : lm.QuantileRegressor(solver='highs',quantile=0.5,alpha=0),
    'DT'   : tree.DecisionTreeRegressor(random_state=0),
    'RF'   : en.RandomForestRegressor(random_state=0)
}

param = {
    'poly__degree'         : [1,2,3],
    'inter__degree'        : [1,2,3],
    'sp__extrapolation'    : ['constant','continue','linear'],
    'sp__knots'            : ['uniform','quantile'],
    'DT__max_depth'        : [2,4,6,8,10],
    'DT__min_samples_split': [5,30,90,200],
    'RF__max_features'     : [1,'sqrt'],
}

struct_lm = [
    'poly_OLS',       'sp_OLS',       'inter_sp_OLS',
    'poly_std_HUBER', 'sp_std_HUBER', 'inter_sp_std_HUBER',
    'poly_std_EN',    'sp_std_EN',    'inter_sp_std_EN',
    'poly_std_LAR',   'sp_std_LAR',   'inter_sp_std_LAR',
]

struct = {
    'lm' : [
        'poly_OLS',       'sp_OLS',       'inter_sp_OLS',
        'poly_std_HUBER', 'sp_std_HUBER', 'inter_sp_std_HUBER',
        'poly_std_EN',    'sp_std_EN',    'inter_sp_std_EN',
        'poly_std_LAR',   'sp_std_LAR',   'inter_sp_std_LAR',
    ],
    'tr' : [
        'DT', 'RF'
    ]
}

def struct_to_estimator(struct):
    struct = struct.split('_')
    model_name = struct[-1]
    preprocess_name = struct[:-1]
    
    pipe = []
    # 在管道中增加预处理的环节
    if len(preprocess_name) > 0:
        for name in preprocess_name:
            if name not in preprocess.keys():
                raise ValueError(f'不存在{name}')
            pipe.append((name, clone(preprocess[name])))
    
    # 在管道中增加模型的环节
    if model_name not in model.keys():
        raise ValueError(f'不存在{model_name}')
    pipe.append((model_name, clone(model[model_name])))
    pipe = Pipeline(pipe)
    return pipe

def struct_to_param(struct):
    struct = struct.split('_')
    model_name = struct[-1]
    preprocess_name = struct[:-1]
    # 生成参数网格
    valid_param = {}
    for param_name,param_space in param.items():
        param_method_name = param_name.split('__')[0]
        if param_method_name in preprocess_name or param_method_name in model_name:
            valid_param[param_name] = param_space
    return valid_param