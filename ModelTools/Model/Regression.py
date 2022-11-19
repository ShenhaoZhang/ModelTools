# 可自定义PIPELINE
# 可依次添加PIPELINE
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV
from tqdm import tqdm

from . import model_config as mc
from ..Metric.Metric import Metric

class Regression:
    def __init__(self,data:pd.DataFrame,col_x:list,col_y:str,col_ts:str=None,ts_freq=None,cv_method='ts',cv_split:int=5) -> None:
        self.data = data 
        self.col_x = col_x
        self.col_y = col_y
        self.col_ts = col_ts
        self.ts_freq = ts_freq
        
        self.train_data, self.test_data = train_test_split(data, test_size=0.3, random_state=0, shuffle=False)
        self.train_x = self.train_data.loc[:,self.col_x]
        self.train_y = self.train_data.loc[:,self.col_y]
        self.test_x  = self.test_data.loc[:,self.col_x]
        self.test_y  = self.test_data.loc[:,self.col_y]
        
        if cv_method == 'ts':
            self.cv_method = TimeSeriesSplit(n_splits=cv_split)
            #TODO 增加其他交叉验证的方法
        
        self.all_model        = {} # 每个value都是GridSearchCV对象
        self.all_param        = {}
        self.all_train_info   = {}
        self.all_train_score  = {} # 该得分将MSE等指标取负值，从而使得该指标越大越好
        self.all_test_predict = {}
        self.best_model_name  = {}
        self.best_model       = None
        self.final_model      = None
        
    def fit(self,add_models:list=None):
        model_struct = mc.base_struct if add_models is None else [*mc.base_struct,*add_models]
        model_struct = list(set(model_struct))
        # TODO 需要增加对add_models的校验
        
        for struct in tqdm(model_struct):
            if isinstance(struct,str):
                name  = struct
                model = get_model_cv(struct=struct,cv=self.cv_method)
            elif isinstance(struct,dict):
                #TODO 此处未测试
                name       = struct['name']
                estimator  = struct['estimator']
                param_grid = struct['param_grid']
                model = get_model_cv(estimator=estimator,param_grid=param_grid,cv=self.cv_method)
            
            if name not in self.all_model.keys():
                model.fit(X=self.train_data.loc[:,self.col_x], y=self.train_data.loc[:,self.col_y])
                self.all_model       [name] = model
                self.all_param       [name] = model.best_params_
                self.all_train_info  [name] = model.cv_results_
                self.all_train_score [name] = model.best_score_
                self.all_test_predict[name] = model.predict(self.test_x)
        
        # 通过各模型在训练集上的得分，初始化最佳模型
        self.best_model_name = max(self.all_train_score,key=self.all_train_score.get)
        self.best_model      = self.all_model.get(self.best_model_name)
        print(f'The best model is {self.best_model_name}')
        
        # 各个模型在测试集上的效果评估
        self.Metric = Metric(
            y_true      = self.test_y.to_numpy(),
            y_pred      = list(self.all_test_predict.values()),
            y_pred_name = list(self.all_test_predict.keys()),
            index       = self.test_data.loc[:,self.col_ts],
            index_freq  = self.ts_freq, 
            hightlight  = [self.best_model_name]
        )

    def add_model(self):
        #TODO 输入模型并训练，并将结果输入至all相关的参数内
        ...
        
    def check_split(self):
        ...
    
    def fit_final_model(self,model='best_model',save_path=None):
        ...
        
def get_model_cv(struct:str,cv,estimator=None,param_grid=None):
    if isinstance(struct,str):
        # 解析模型结构字符
        estimator  = mc.struct_to_estimator(struct)
        param_grid = mc.struct_to_param(struct)  
    else:
        estimator  = estimator
        param_grid = param_grid
        # TODO 输入模型，输出GridSearchCV中可以作为estimator的对象，此处的代码可以是该对象规则的校验
    
    # 网格搜索的交叉验证方法
    search = GridSearchCV(
        estimator          = estimator,
        param_grid         = param_grid,
        refit              = True,
        cv                 = cv,
        return_train_score = True,
        n_jobs             = -1,
        scoring            = 'neg_mean_squared_error'
    )
    return search


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=100)
    x2 = rng.normal(size=100)
    y = rng.normal(loc=np.sin(x1) + np.cos(x2),scale=1)
    df = pd.DataFrame(data={'x1':x1,'x2':x2,'y':y})
    
    print(df)
    
    m = Regression(data=df,col_x=['x1','x2'],col_y='y')
    m.fit()
    print(m.all_train_score)