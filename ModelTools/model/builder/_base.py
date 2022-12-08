from sklearn.base import clone
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    KFold
)
from sklearn import linear_model as lm
from sklearn import tree
from sklearn import ensemble as en
from sklearn import preprocessing as pr
from sklearn import metrics

#TODO 检查如何能按工况进行分组抽样

class BaseBuilder:
    def __init__(
        self,
        cv_method,
        cv_split,
        cv_shuffle,
        cv_score:str,
    ) -> None:

        self.cv_method  = cv_method
        self.cv_split   = cv_split
        self.cv_shuffle = cv_shuffle
        self.cv_score   = cv_score

        self.preprocess = None
        self.param      = None
        self.model      = None
        self.struct     = None
        
        self.cv    = self.get_cv(self.cv_method,self.cv_split,self.cv_shuffle)
        self.score = self.get_cv_score(self.cv_score)
    
    @staticmethod
    def get_cv(cv_method,cv_split,cv_shuffle):
        if cv_method == 'ts':
            cv = TimeSeriesSplit(n_splits=cv_split)
        elif cv_method == 'kfold':
            cv_random_state = 0 if cv_shuffle == True else None
            cv = KFold(n_splits=cv_split,shuffle=cv_shuffle,random_state=cv_random_state)
        return cv 
    
    @staticmethod
    def get_cv_score(score_name):
        score = {}
        score = score.get(score_name)
        return score

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
            if param_method_name in preprocess_name or param_method_name == model_name:
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
    
    def get_model_cv(self,struct:str=None,estimator=None,param_grid=None):
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
            cv                 = self.cv,
            return_train_score = True,
            n_jobs             = -1,
            scoring            = self.score
        )
        return model_cv
    



        
