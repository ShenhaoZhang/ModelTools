import sys 
import os 
import warnings
import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    KFold
)
from tqdm import tqdm
from tabulate import tabulate
import altair as alt 
alt.data_transformers.disable_max_rows()

from .reg_config import RegressionConfig
from ..data.data import Data
from ..metric.metric import Metric
from ..explain.explain import Explain
from ..plot.corr import scatter

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::RuntimeWarning')

class Regression:
    def __init__(self, data:Union[pd.DataFrame,dict], col_x:Union[str,list], col_y:str, col_ts:str = None, ts_freq = None, 
                 split_test_size:float = 0.3, split_shuffle = False, cv_method:str = 'kfold', cv_split:int = 5, exp_model:bool = True) -> None:
        self.data      = data
        self.col_x     = col_x if isinstance(col_x,list) else [col_x]
        self.col_y     = col_y
        self.exp_model = exp_model
        
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
            self.split_test_size = round(len(self.test_data) / len(self.data),2)
            self.split_shuffle = False
            
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
        self.reg_config = RegressionConfig() # 回归模型管道的配置
        
        self.all_model         = {}  # 每个value都是GridSearchCV对象
        self.all_param         = {}
        self.all_cv_results    = {}
        self.all_train_score   = {}  # 该得分将MSE等指标取负值, 从而使得该指标越大越好
        self.all_train_predict = {}
        self.all_test_predict  = {}
        self.best_model_name   = {}
        self.best_model        = None
        self.best_model_param  = None
        self.final_model       = None
        self.final_model_name  = None
        
        # 数据的探索性分析
        self.Data = Data(
            data   = {'train':self.train_data,'test':self.test_data},
            col_x  = self.col_x,
            col_y  = self.col_y,
            col_ts = self.col_ts
        )  
        self.MetricTrain = None  # 模型在训练集上的效果评价
        self.MetricTest  = None  # 模型在测试集上的效果评价
        self.MetricFinal = None  # 最终模型在整个数据集上的效果评价
        self.ExpMod      = None  # 基于模型的解释
        self.ExpResid    = None  # 基于测试集残差的解释
        self.ExpFinal    = None  # 基于最终模型的解释
    
    def fit(self, base = ['lm'], best_model:str = 'auto', best_model_only:bool = False, add_models:list = None, update_param:dict = None, 
            print_result:bool = True):
        """
        模型拟合

        Parameters
        ----------
        base : list, optional
            基准模型库, by default ['lm']
            lm: 线性模型
            tr: 树模型
        add_models : list, optional
            在基准模型库中增加模型, 有两种定义方式, 可混合使用, by default None
            方法一: 库中通过结构化字符的方式定义的模型, 例如: 'poly_OLS'  
            方法二: 通过字典定义模型的名字、sklearn中的Pipeline和超参数, 例如{'name':'OLS','estimator':Pipeline(...),'param_grid':{'poly__degree':[1,2]}}  
        best_model : str, optional
            指定最佳模型, 训练后的最佳模型为该模型, by default 'auto'
        best_model_only : bool, optional
            仅拟合指定的最佳模型, by default False
        update_param : dict, optional
            覆盖配置中的某个超参数, 字典中的value可以是单个值或列表, by default None
            例如: {'poly__degree':3} 或 {'poly__degree':[4,5,6]}
        print_result : bool, optional
            打印模型结果, by default True
        """
        if (best_model_only is True) and (best_model == 'auto'):
            raise Exception('当best_model_only为True时, best_model不能为auto')
        
        if update_param is not None:
            self.reg_config.update_param(update_param)
            update_param_name = [param.split('__')[0] for param in update_param.keys()]
            
        model_struct = []
        for bases_type in base:
            model_struct += self.reg_config.struct.get(bases_type)
        
        # 在基础模型上增加模型
        if add_models is not None:
            add_models = add_models if isinstance(add_models,list) else [add_models]
            model_struct += add_models
            model_struct = list(set(model_struct))
        # TODO 需要增加对add_models的校验
        
        for struct in tqdm(model_struct):
            # 解析字符
            if isinstance(struct,str):
                name  = struct
                model = self.reg_config.get_model_cv(struct=struct,cv_method=self.cv_method)
            elif isinstance(struct,dict):
                #TODO 此处未测试
                name       = struct['name']
                estimator  = struct['estimator']
                param_grid = struct['param_grid']
                model = self.reg_config.get_model_cv(estimator=estimator,param_grid=param_grid,cv_method=self.cv_method)
            
            # 判断是否需要重新拟合
            is_contain_param = any([p_name in name for p_name in update_param_name]) if update_param is not None else False # struct中是否包含了任意一个update_param
            is_best_model    = name == best_model             # struct是否是指定的最佳模型
            is_already_fit   = name in self.all_model.keys()  # struct是否已拟合
            must_refit       = is_contain_param and is_best_model if best_model_only else is_contain_param
            if not must_refit:
                if best_model_only and not is_best_model:
                    continue
                if is_already_fit:
                    continue
            
            model.fit(X=self.train_data.loc[:,self.col_x], y=self.train_data.loc[:,self.col_y])
            self.all_model         [name] = model
            self.all_param         [name] = model.best_params_
            self.all_cv_results    [name] = model.cv_results_
            self.all_train_predict [name] = model.predict(self.train_x)
            self.all_train_score   [name] = model.best_score_
            self.all_test_predict  [name] = model.predict(self.test_x)
        
        if best_model == 'auto':
            # 通过各模型在训练集上的得分, 初始化最佳模型
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
        
        if self.exp_model == True:
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
    
    def fit_final_model(self,model='best_model',print_result=True):
        self.final_model_name = self.best_model_name if model == 'best_model' else model
        self.final_model = clone(self.all_model.get(self.final_model_name).best_estimator_) # 此处保留了超参数
        self.final_model.fit(X=self.data.loc[:,self.col_x], y=self.data.loc[:,self.col_y].to_numpy())
        
        # 最终模型在全部数据上的表现
        self.MetricFinal = Metric(
            y_true      = self.data.loc[:,self.col_y].to_numpy(),
            y_pred      = [self.predict()],
            y_pred_name = [self.final_model_name],
            index       = None if self.col_ts is None else self.data.loc[:,self.col_ts],
            index_freq  = self.ts_freq, 
        )
        if self.exp_model == True:
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
                f'Final Model : {self.final_model_name} \n'
                f"{tabulate(final_model_matric,headers=final_model_matric.columns)}"
            )
            print(message)
        
        return self
    
    def save_final_model(self,path) -> None:
        # TODO 更新保存模型的方法
        # https://scikit-learn.org/stable/model_persistence.html#interoperable-formats
        self.__check_model_status(type='final')
        pickle.dump(obj=self.final_model,file=open(path,'wb'))
    
    def predict(self,new_data=None) -> np.ndarray:
        self.__check_model_status(type='final')
        
        if new_data is None:
            new_data = self.data.loc[:,self.col_x]
        else:
            new_data = new_data.loc[:,self.col_x]
            
        pred = self.final_model.predict(new_data)
        return pred

    def predict_ci(self,new_data:pd.DataFrame=None,n_bootstrap=1000,alpha=0.05) -> dict:
        self.__check_model_status(type='final')
        
        if new_data is None:
            new_data = self.data.loc[:,self.col_x]
        else:
            new_data = new_data.loc[:,self.col_x]
            
        sample = np.empty(shape=[len(new_data),n_bootstrap])
        for i in tqdm(range(n_bootstrap)):
            data = self.data.sample(frac=1,replace=True)
            mod = clone(self.final_model)
            mod.fit(X=data.loc[:,self.col_x],y=data.loc[:,self.col_y])
            sample[:,i] = mod.predict(new_data)
            
        low    = np.quantile(sample,q=alpha/2,axis=1)
        median = np.quantile(sample,q=0.5,axis=1)
        high   = np.quantile(sample,q=1-alpha/2,axis=1)
        result = {'down':low,'median':median,'high':high}
        
        return result
            
        
        
    def check_cv_split(self,plot=True):
        # TODO 拟合的时间、预测效果、可视化
        ...
    
    
    def check_novelty(self,method,new_data=None,return_score=False):
        #TODO 分两种情况 1. 基于train data 查看 test data  2.基于 all data 查看 new_data
        if new_data is None:
            if len(self.all_test_predict)==0:
                raise Exception('需要先拟合模型')
            data = pd.DataFrame({
                'score'       : self.Data.get_novelty_score(method=method),
                'abs_residual': np.abs(self.test_y.to_numpy() - self.all_test_predict[self.best_model_name])
            })
            plot = scatter(
                data      = data,
                x         = 'score',
                y         = 'abs_residual',
                add_hline = self.MetricTrain.get_metric().at[self.best_model_name,'MAE'],
            )
        else:
            # TODO 调整Novelty
            from ..tools.novelty import Novelty
            self.__check_model_status(type='final')
            score = Novelty(
                train_x=self.data.loc[:,self.col_x],
                test_x=new_data.loc[:,self.col_x]
            ).get_score(method=method)
            data = pd.DataFrame({
                'score'       : score,
                'abs_residual': np.abs(new_data.loc[:,self.col_y].to_numpy() - self.predict(new_data.loc[:,self.col_x]))
            })
            plot = scatter(
                data      = data,
                x         = 'score',
                y         = 'abs_residual',
                add_hline = self.MetricFinal.get_metric().at[self.final_model_name,'MAE'],
            )
        
        if not return_score:
            return plot
        else:
            return data.score.to_numpy()
    
    def __check_model_status(self,type='final'):
        if type is 'final':
            if self.final_model is None:
                raise Exception('模型需要先fit_final_model')
        # elif type == 'test':
        #     if self.