import sys 
import os 
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    KFold,
    GridSearchCV
)
from tqdm import tqdm
from tabulate import tabulate

from . import model_config as mc
from ..Metric.Metric import Metric
from ..Explain.Explain import Explain
from ..tools.Novelty import Novelty

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::RuntimeWarning')

class Regression:
    def __init__(self, data:pd.DataFrame, col_x:list, col_y:str, col_ts:str = None, ts_freq = None, 
                 split_test_size:float = 0.3, split_shuffle = False, cv_method:str = 'kfold', cv_split:int = 5 ) -> None:
        self.data    = data
        self.col_x   = col_x if isinstance(col_x,list) else [col_x]
        self.col_y   = col_y if isinstance(col_y,str) else col_y[0]
        
        # 分割数据集
        if isinstance(self.data,pd.DataFrame):
            self.split_test_size = split_test_size
            self.split_shuffle   = split_shuffle
            self.train_data, self.test_data = train_test_split(data, test_size=split_test_size, random_state=0, shuffle=split_shuffle)
        elif isinstance(self.data,dict):
            self.train_data = self.data.get('train')
            self.test_data = self.data.get('test')
            if set(self.train_data.columns) != set(self.test_data.columns):
                raise Exception('训练集和测试集的字段名称不匹配')
            self.data = pd.concat(list(self.data.values()),axis=0)
        self.train_x = self.train_data.loc[:,self.col_x]
        self.train_y = self.train_data.loc[:,self.col_y]
        self.test_x  = self.test_data.loc[:,self.col_x]
        self.test_y  = self.test_data.loc[:,self.col_y]

        # 定义时间变量
        if col_ts is not None:
            if col_ts not in self.data.columns:
                raise Exception(f'data中缺失{col_ts}')
            if ts_freq is None:
                raise Exception('缺失ts_freq参数')
            self.data.loc[:,col_ts] = pd.DatetimeIndex(self.data.loc[:,col_ts])
            self.data = self.data.sort_values(by=col_ts,ascending=True)
        self.col_ts  = col_ts
        self.ts_freq = ts_freq
        
        # 定义交叉验证的方法
        if cv_method == 'ts':
            self.cv_method = TimeSeriesSplit(n_splits=cv_split)
        elif cv_method == 'kfold':
            self.cv_method = KFold(n_splits=cv_split, shuffle=True, random_state=0)
        
        self.all_model         = {}  # 每个value都是GridSearchCV对象
        self.all_param         = {}
        self.all_cv_results    = {}
        self.all_train_score   = {}  # 该得分将MSE等指标取负值，从而使得该指标越大越好
        self.all_train_predict = {}
        self.all_test_predict  = {}
        self.best_model_name   = {}
        self.best_model        = None
        self.best_model_param  = None
        self.final_model       = None
        self.final_predict     = None 
        
        self.Data        = None  # 数据的探索性分析
        self.MetricTrain = None  # 模型在训练集上的效果评价
        self.MetricTest  = None  # 模型在测试集上的效果评价
        self.MetricFinal = None
        self.ExpMod      = None  # 基于模型的解释
        self.ExpResid    = None  # 基于测试集残差的解释
        self.ExpFinal    = None
        self.Novelty     = None  #TODO 放到Data模块中
    
    def split_data(self):
        return self
        
    @staticmethod   
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
    
    def fit(self,base=['lm'],add_models:list=None,best_model:str='auto',best_model_only:bool=False,print_result:bool=True):
        
        base_struct = []
        for s_type in base:
            base_struct += mc.struct.get(s_type)
        
        add_models   = [add_models] if (add_models is not None) and (not isinstance(add_models,list)) else add_models
        model_struct = base_struct if add_models is None else [*base_struct,*add_models]
        model_struct = list(set(model_struct))
        # TODO 需要增加对add_models的校验
        
        for struct in tqdm(model_struct):
            if isinstance(struct,str):
                name  = struct
                model = self.get_model_cv(struct=struct,cv=self.cv_method)
            elif isinstance(struct,dict):
                #TODO 此处未测试
                name       = struct['name']
                estimator  = struct['estimator']
                param_grid = struct['param_grid']
                model = self.get_model_cv(estimator=estimator,param_grid=param_grid,cv=self.cv_method)
            
            if (best_model != 'auto' ) and (best_model_only is True) and (name != best_model):
                continue
            
            if name in self.all_model.keys():
                continue
            
            model.fit(X=self.train_data.loc[:,self.col_x], y=self.train_data.loc[:,self.col_y])
            self.all_model         [name] = model
            self.all_param         [name] = model.best_params_
            self.all_cv_results    [name] = model.cv_results_
            self.all_train_predict [name] = model.predict(self.train_x)
            self.all_train_score   [name] = model.best_score_
            self.all_test_predict  [name] = model.predict(self.test_x)
        
        if best_model == 'auto':
            # 通过各模型在训练集上的得分，初始化最佳模型
            # TODO 检查此处的逻辑 
            self.best_model_name = max(self.all_train_score,key=self.all_train_score.get)
        else:
            self.best_model_name = best_model
        self.best_model       = self.all_model.get(self.best_model_name)
        self.best_model_param = self.all_param.get(self.best_model_name)
            
        
        # 各个模型在训练集上的效果评估
        self.MetricTrain = Metric(
            y_true      = self.train_y.to_numpy(),
            y_pred      = list(self.all_train_predict.values()),
            y_pred_name = list(self.all_train_predict.keys()),
            index       = None if self.col_ts is None else self.train_data.loc[:,self.col_ts],
            index_freq  = self.ts_freq, 
            highlight   = {self.best_model_name:'Best_Model(CV)'}
        )
        
        # 各个模型在测试集上的效果评估
        self.MetricTest = Metric(
            y_true      = self.test_y.to_numpy(),
            y_pred      = list(self.all_test_predict.values()),
            y_pred_name = list(self.all_test_predict.keys()),
            index       = None if self.col_ts is None else self.test_data.loc[:,self.col_ts],
            index_freq  = self.ts_freq, 
            highlight   = {self.best_model_name:'Best_Model(CV)'}
        )
        
        # 解释模型
        self.ExpMod = Explain(
            model      = self.best_model,
            model_type = 'regression',
            data_x     = self.test_x,
            data_y     = self.test_y
        )
        
        # 解释测试集上的预测误差
        #TODO 考虑增加样本序号
        abs_resid = np.abs(self.test_y - self.best_model.predict(self.test_x))
        resid_model = clone(self.best_model).fit(self.test_x,abs_resid)
        self.ExpResid = Explain(
            model      = resid_model,
            model_type = 'regression',
            data_x     = self.test_x,
            data_y     = abs_resid
        )
        
        self.Novelty = Novelty(
            train_x = self.train_x,
            test_x  = self.test_x
        )

        # 打印结果
        if print_result == True:
            best_model_metric = pd.concat([
                self.MetricTrain.get_metric().loc[[self.best_model_name]],
                self.MetricTest.get_metric().loc[[self.best_model_name]]
            ])
            best_model_metric.index = ['Train','Test']
            best_model_param = ', '.join([f'{param_name}={param_value}' for param_name,param_value in self.best_model_param.items()])
            message = (
                f"Best Model(CV)   : {self.best_model_name} \n"
                f"Hyperparameters  : {best_model_param} \n"
                f"Train Test Split : test_size={self.split_test_size}, shuffle={self.split_shuffle}, random_state=0 \n"
                f"Cross Validation : {str(self.cv_method)} \n \n"
                f"{tabulate(best_model_metric.round(4),headers=best_model_metric.columns)} \n \n"
                "Regression.MetricTrain : 模型在训练集上的效果评价 \n"
                "Regression.MetricTest  : 模型在测试集上的效果评价 \n"
                "Regression.ExpResid    : 基于模型的残差解释 \n"
                "Regression.ExpMod      : 基于模型的特征解释 \n"
            )
            print(message)
        
        return self

    def check_cv_split(self,plot=True):
        # TODO 拟合的时间、预测效果、可视化
        ...
    
    
    def check_novelty(self):
        import plotnine as gg
        plot = (
            gg.qplot(
                x = self.Novelty.get_score(),
                y = np.abs(self.test_y.to_numpy() - self.all_test_predict[self.best_model_name]),
                geom='point'
            )+gg.geom_smooth(method='lm')
            +gg.geom_hline(yintercept=self.MetricTrain.get_metric().at[self.best_model_name,'MAE'],color='green')
        )
        return plot
    
    def fit_final_model(self,model='best_model',print_result=True):
        final_model_name = self.best_model_name if model == 'best_model' else model
        self.final_model = clone(self.all_model.get(final_model_name).best_estimator_) # 此处保留了超参数
        self.final_model.fit(X=self.data.loc[:,self.col_x], y=self.data.loc[:,self.col_y].to_numpy())
        self.final_predict = self.final_model.predict(self.data.loc[:,self.col_x])
        
        # 最终模型在全部数据上的表现
        self.MetricFinal = Metric(
            y_true      = self.data.loc[:,self.col_y].to_numpy(),
            y_pred      = [self.final_predict],
            y_pred_name = [final_model_name],
            index       = None if self.col_ts is None else self.data.loc[:,self.col_ts],
            index_freq  = self.ts_freq, 
        )
        self.ExpFinal = Explain(
            model      = self.final_model,
            model_type = 'regression',
            data_x     = self.data.loc[:,self.col_x],
            data_y     = self.data.loc[:,self.col_y].to_numpy()
        )
        
        if print_result:
            final_model_matric = self.MetricFinal.get_metric().round(4)
            final_model_matric.index = ['Train & Test']
            message = (
                f'Final Model : {final_model_name} \n'
                f"{tabulate(final_model_matric,headers=final_model_matric.columns)}"
            )
            print(message)
        
        return self
    
    def predict(self,x):
        #TODO 置信区间的预测
        pred = self.final_model.predict(x)
        return pred
