from sklearn.base import clone
from sklearn import linear_model as lm
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
    'HUBER': lm.HuberRegressor(max_iter=500),
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