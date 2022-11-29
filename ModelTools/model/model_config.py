from sklearn.base import clone
from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble as en
from sklearn import preprocessing as pr
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV

class RegressionConfig:
    def __init__(self) -> None:
        
        self.preprocess = {
            'std'  : pr.StandardScaler(),
            'poly' : pr.PolynomialFeatures(),
            'inter': pr.PolynomialFeatures(interaction_only=True),
            'sp'   : pr.SplineTransformer()
        }
        
        self.model = {
            'OLS'  : lm.LinearRegression(),
            'LAR'  : lm.Lars(normalize=False),
            'HUBER': lm.HuberRegressor(),
            'EN'   : lm.ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,.99,1],random_state=0),
            'QR'   : lm.QuantileRegressor(solver='highs',quantile=0.5,alpha=0),
            'DT'   : tree.DecisionTreeRegressor(random_state=0),
            'RF'   : en.RandomForestRegressor(random_state=0)
        }
        
        self.param = {
            'poly__degree'         : [1,2,3],
            'inter__degree'        : [1,2,3],
            'sp__extrapolation'    : ['constant','continue','linear'],
            'sp__knots'            : ['uniform','quantile'],
            'DT__max_depth'        : [2,4,6,8,10],
            'DT__min_samples_split': [5,30,90,200],
            'RF__max_features'     : [1,'sqrt'],
        }
        
        self.struct = {
            'lm' : [
                'OLS',        'poly_OLS',       'sp_OLS',       'inter_sp_OLS',        
                'std_HUBER',  'poly_std_HUBER', 'sp_std_HUBER', 'inter_sp_std_HUBER',  
                'std_EN',     'poly_std_EN',    'sp_std_EN',    'inter_sp_std_EN',     
                'std_LAR',    'poly_std_LAR',   'sp_std_LAR',   'inter_sp_std_LAR',    
            ],
            'tr' : [
                'DT', 'RF'
            ]
        }
    
    @classmethod
    def translate_struct(cls,struct):
        struct_decomp   = struct.split('_')
        model_name      = struct_decomp[-1]
        preprocess_name = struct_decomp[:-1]
        return preprocess_name,model_name
        
    def struct_to_estimator(self,struct) -> Pipeline:
        preprocess_name, model_name = self.translate_struct(struct)
        pipe = []
        # 在管道中增加预处理的环节
        if len(preprocess_name) > 0:
            for name in preprocess_name:
                if name not in self.preprocess.keys():
                    raise ValueError(f'不存在{name}')
                pipe.append((name, clone(self.preprocess[name])))
        
        # 在管道中增加模型的环节
        if model_name not in self.model.keys():
            raise ValueError(f'不存在{model_name}')
        pipe.append((model_name, clone(self.model[model_name])))
        pipe = Pipeline(pipe)
        return pipe

    def struct_to_param(self,struct) -> dict:
        preprocess_name, model_name = self.translate_struct(struct)
        # 生成参数网格
        # 输入指定超参数空间，替代默认值
        valid_param = {}          
        for param_name,param_space in self.param.items():
            param_method_name = param_name.split('__')[0]
            if param_method_name in preprocess_name or param_method_name in model_name:
                valid_param[param_name] = param_space
        return valid_param

    def update_param(self,update_param:dict) -> None:
        for param_name,param_space in update_param.items():
            if param_name not in self.param.keys():
                raise Exception(f'不存在{param_name}')
            if isinstance(param_space,list):
                self.param.update({param_name:param_space})
            else:
                self.param.update({param_name:[param_space]})
    
    def get_model_cv(self,cv_method,struct:str=None,estimator=None,param_grid=None):
        if struct is not None:
            estimator = self.struct_to_estimator(struct)
            param_grid = self.struct_to_param(struct)
        elif (estimator is not None) and (param_grid is not None):
            estimator = estimator
            param_grid = param_grid
            # TODO 输入模型, 输出GridSearchCV中可以作为estimator的对象, 此处的代码可以是该对象规则的校验
        else:
            raise Exception('必须输入cv_method或estimator和param_grid')
        
        model_cv = GridSearchCV(
            estimator          = estimator,
            param_grid         = param_grid,
            refit              = True,
            cv                 = cv_method,
            return_train_score = True,
            n_jobs             = -1,
            scoring            = 'neg_mean_squared_error'
        )
        return model_cv

if __name__ == '__main__':
    rc = RegressionConfig.translate_struct('poly_OLS')
    print(rc)