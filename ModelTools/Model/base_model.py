from sklearn.base import clone
from sklearn.pipeline import Pipeline 
from sklearn import linear_model as lm
from sklearn import preprocessing as pr
from sklearn.decomposition import PCA 

preprocess = {
    'poly':pr.PolynomialFeatures(),
    'inter':pr.PolynomialFeatures(interaction_only=True),
    'sp':pr.SplineTransformer()
}

param = {
    'poly__degree':[1,2,3],
    'inter__degree':[1,2,3]
}

model = {
    'ols':lm.LinearRegression(),
    'huber':lm.HuberRegressor()
}

def get_model(struct:str):
    # 解析模型结构字符
    struct = struct.split('_')
    model_name = struct[-1]
    preprocess_name = struct[:-1]
    
    pipe = []
    # 在管道中增加预处理的模块
    if len(preprocess_name) > 0:
        for name in preprocess_name:
            if name not in preprocess.keys():
                raise ValueError(f'不存在{name}')
            pipe.append((name, clone(preprocess[name])))
    
    # 在管道中增加模型的模块
    if model_name not in model.keys():
        raise ValueError(f'不存在{model_name}')
    pipe.append((model_name, clone(model[model_name])))
    
    pipe = Pipeline(pipe)
    
    return pipe

if __name__ == '__main__':
    print(get_model('inter_sp_huber'))