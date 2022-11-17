# 可自定义PIPELINE
# 可依次添加PIPELINE
import numpy as np
import pandas as pd
import plotnine as gg
from sklearn.base import clone
from sklearn.pipeline import Pipeline 
from sklearn import linear_model as lm
from sklearn import preprocessing as pr
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split,TimeSeriesSplit,GridSearchCV

from . import model_tools as tools

class Model:
    def __init__(self,data:pd.DataFrame,col_x:list,col_y:str) -> None:
        self.data = data 
        self.col_x = col_x
        self.col_y = col_y
        
        self.train_data, self.test_data = train_test_split(data, test_size=0.3, random_state=0, shuffle=False)
        self.train_x = self.train_data.loc[:,self.col_x]
        self.train_y = self.train_data.loc[:,self.col_y]
        self.test_x = self.test_data.loc[:,self.col_x]
        self.test_y = self.test_data.loc[:,self.col_y]
        
        self.cv_method = None
        self.all_model = None
        self.all_test_predict = None
        
    def fit(self,add_struct=None):
        base_struct = ['poly_ols','inter_sp_ols']
        #TODO struct校验
        model_struct = base_struct if add_struct is None else base_struct + add_struct
        
        self.all_model = {}
        self.all_test_predict = {}
        self.all_train_info = {}
        self.cv_method = TimeSeriesSplit(n_splits=5)
        
        for struct in model_struct:
            model = get_pipeline(struct=struct,param=tools.param,cv=self.cv_method)
            model.fit(X=self.train_data.loc[:,self.col_x],
                      y=self.train_data.loc[:,self.col_y])
            self.all_model[struct] = model
            self.all_train_info[struct] = model.cv_results_
            self.all_test_predict[struct] = model.predict(self.test_x)
    
    def check_split(self):
        ...
    
    def add_model(self):
        ...
    
    def choose_model(self):
        ...
    
def get_pipeline(struct:str,param:dict,cv):
    # 解析模型结构字符
    struct = struct.split('_')
    model_name = struct[-1]
    preprocess_name = struct[:-1]
    
    pipe = []
    # 在管道中增加预处理的环节
    if len(preprocess_name) > 0:
        for name in preprocess_name:
            if name not in tools.preprocess.keys():
                raise ValueError(f'不存在{name}')
            pipe.append((name, clone(tools.preprocess[name])))
    
    # 在管道中增加模型的环节
    if model_name not in tools.model.keys():
        raise ValueError(f'不存在{model_name}')
    pipe.append((model_name, clone(tools.model[model_name])))
    pipe = Pipeline(pipe)
    
    # 生成参数网格
    valid_param = {}
    for param_name,param_space in param.items():
        param_method_name = param_name.split('__')[0]
        if param_method_name in preprocess_name or param_method_name in model_name:
            valid_param[param_name] = param_space
    
    # 网格搜索的交叉验证方法
    search = GridSearchCV(
        estimator=pipe,
        param_grid=valid_param,
        refit=True,
        cv=cv,
        return_train_score=True
    )
    return search


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=100)
    x2 = rng.normal(size=100)
    y = rng.normal(loc=np.sin(x1) + np.cos(x2),scale=1)
    df = pd.DataFrame(data={'x1':x1,'x2':x2,'y':y})
    
    print(df)
    
    m = Model(data=df,col_x=['x1','x2'],col_y='y')
    m.fit()
    print(m.all)