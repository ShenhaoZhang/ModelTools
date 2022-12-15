#TODO 增加训练和测试上模型的效果比较
#TODO 如果工况分布不均匀 且 成块出现，不应split_shuffle为False，这样交叉验证的结果必然不好，应该分组抽样，cv的shuffle也应该分组抽样多特征如何分组？首先PCA？
#TODO 预测区间和置信区间的可视化

import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tabulate import tabulate

from ...data.data import Data
from ...explain.model_agnostic import MaExplain
from ...tools.novelty import Novelty
from ...plot.corr_scatter import corr_scatter

class BaseModel:
    def __init__(
        self, 
        data           :Union[pd.DataFrame,dict], 
        col_x          :Union[str,list], 
        col_y          :str, 
        col_ts         :str   = None,
        ts_freq        :str   = None,
        split_test_size:float = 0.3,
        split_shuffle  :bool  = False,
        cv_method      :str   = 'kfold',
        cv_split       :int   = 5,
        cv_shuffle     :bool  = False,
        cv_score       :str   = 'mse',
        exp_model      :bool  = True
    ) -> None:
        
        self.data            = data
        self.col_x           = col_x if isinstance(col_x,list) else [col_x]
        self.col_y           = col_y
        self.cv_method       = cv_method
        self.cv_split        = cv_split
        self.cv_shuffle      = cv_shuffle
        self.cv_score        = cv_score
        self._exp_model      = exp_model
        self.col_ts          = col_ts
        self.ts_freq         = ts_freq
        self.split_test_size = split_test_size
        self.split_shuffle   = split_shuffle
        
        self.all_model         = {}  # 每个value都是GridSearchCV对象
        self.all_param         = {}
        self.all_cv_results    = {}
        self.all_train_score   = {}  # 该得分将MSE等指标取负值, 从而使得该指标越大越好
        self.all_train_predict = {}
        self.all_test_predict  = {}
        
        self.best_model            = None
        self.best_model_name       = None
        self.best_model_param      = None
        self.best_model_test_resid = None
        
        self.final_model       = None
        self.final_model_name  = None
        self.final_model_param = None
        self.final_model_resid = None
        
        self.MetricTrain = None  # 模型在训练集上的效果评价
        self.MetricTest  = None  # 模型在测试集上的效果评价
        self.MetricFinal = None  # 最终模型在整个数据集上的效果评价
        self.ExpTrain    = None  # 基于模型的解释
        self.ExpFinal    = None  # 基于最终模型的解释
        
        self._init_data()
    
    def _init_data(self):
        # 分割数据集
        if isinstance(self.data,pd.DataFrame):
            self.split_test_size = self.split_test_size
            self.split_shuffle   = self.split_shuffle
            self.train_data, self.test_data = train_test_split(self.data, test_size=self.split_test_size, random_state=0, shuffle=self.split_shuffle)
        elif isinstance(self.data,dict):
            self.train_data = self.data.get('train')
            self.test_data = self.data.get('test')
            #TODO 用for循环变量，从而方便看哪个字段缺失
            if set(self.train_data.columns) != set(self.test_data.columns):
                raise Exception('训练集和测试集的字段名称不匹配')
            self.data = pd.concat(list(self.data.values()),axis=0)
            self.split_test_size = round(len(self.test_data) / len(self.data),2)
            self.split_shuffle = False
        
        na_sample = self.data.isna().any(axis=1).sum()
        if na_sample > 0:
            self.data = self.data.dropna()
            print(f'剔除数据中{na_sample}个含缺失值的样本，占总数据量的{round(na_sample/len(self.data),4)*100}%')
        
        self.train_x = self.train_data.loc[:,self.col_x]
        self.train_y = self.train_data.loc[:,self.col_y]
        self.test_x  = self.test_data.loc[:,self.col_x]
        self.test_y  = self.test_data.loc[:,self.col_y]
        self.data_x  = self.data.loc[:,self.col_x]
        self.data_y  = self.data.loc[:,self.col_y]
        
        # 定义时间变量
        if self.col_ts is not None:
            if self.col_ts not in self.data.columns:
                raise Exception(f'data中缺失{self.col_ts}')
            if self.ts_freq is None:
                raise Exception('缺失ts_freq参数')
            self.data.loc[:,self.col_ts] = pd.DatetimeIndex(self.data.loc[:,self.col_ts])
            self.data = self.data.sort_values(by=self.col_ts,ascending=True)
        
        # 数据的探索性分析
        self.Data = Data(
            data   = {'train':self.train_data,'test':self.test_data},
            col_x  = self.col_x,
            col_y  = self.col_y,
            col_ts = self.col_ts
        ) 
    
    #TODO 考虑放到fit中
    def _init_model(self,Builder,Metric,metric_kwargs:dict=None):
        # 配置模型的Pipeline
        self._builder = Builder(
            cv_score   = self.cv_score,
            cv_method  = self.cv_method,
            cv_shuffle = self.cv_shuffle,
            cv_split   = self.cv_split
        ) 
        
        self._metric = Metric
        self._metric_kwargs = metric_kwargs if metric_kwargs is not None else {}
        
    def fit(
        self, 
        base            :Union[str,list] = 'lm',
        best_model      :str             = 'auto',
        best_model_only :bool            = False,
        add_models      :list            = None,
        update_param    :dict            = None,
        print_result    :bool            = True
    ):
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
            self._builder.update_param(update_param)
            update_param_name = [param.split('__')[0] for param in update_param.keys()]
            
        model_struct = []
        base = base if isinstance(base,list) else [base]
        for bases_type in base:
            if bases_type not in self._builder.struct.keys():
                raise ValueError(f'base有误: {bases_type}')
            model_struct += self._builder.struct.get(bases_type)
        
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
                model = self._builder.get_model_cv(struct=struct)
            elif isinstance(struct,dict):
                #TODO 此处未测试
                name       = struct['name']
                estimator  = struct['estimator']
                param_grid = struct['param_grid']
                model = self._builder.get_model_cv(estimator=estimator,param_grid=param_grid)
            
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
            # 若模型无效时（预测结果为NaN），不保存模型的结果
            train_predict = model.predict(self.train_x)
            if not np.any(np.isnan(train_predict)):
                self.all_model         [name] = model
                self.all_param         [name] = model.best_params_
                self.all_cv_results    [name] = model.cv_results_
                self.all_train_predict [name] = train_predict
                self.all_train_score   [name] = model.best_score_
                self.all_test_predict  [name] = model.predict(self.test_x)
        
        if best_model == 'auto':
            # 通过各模型在训练集上的得分, 初始化最佳模型
            # 此处得分的定义来源于self.cv_score
            self.best_model_name = max(self.all_train_score,key=self.all_train_score.get)
        else:
            self.best_model_name = best_model
        self.best_model            = self.all_model.get(self.best_model_name)
        self.best_model_param      = self.all_param.get(self.best_model_name)
        self.best_model_test_resid = self.test_y.to_numpy() - self.best_model.predict(self.test_x)
        
        # 各个模型在训练集上的效果评估
        self.MetricTrain = self._metric(
            y_true      = self.train_y.to_numpy(),
            y_pred      = list(self.all_train_predict.values()),
            y_pred_name = list(self.all_train_predict.keys()),
            index       = None if self.col_ts is None else self.train_data.loc[:,self.col_ts],
            index_freq  = self.ts_freq, 
            highlight   = {self.best_model_name:'Best_Model(CV)'},
            **self._metric_kwargs
        )
        
        # 各个模型在测试集上的效果评估
        self.MetricTest = self._metric(
            y_true      = self.test_y.to_numpy(),
            y_pred      = list(self.all_test_predict.values()),
            y_pred_name = list(self.all_test_predict.keys()),
            index       = None if self.col_ts is None else self.test_data.loc[:,self.col_ts],
            index_freq  = self.ts_freq, 
            highlight   = {self.best_model_name:'Best_Model(CV)'},
            **self._metric_kwargs
        )
        
        if self._exp_model == True:
            # 解释模型
            self.ExpTrain = MaExplain(
                model      = self.best_model,
                model_type = 'regression',
                data_x     = self.test_x,
                data_y     = self.test_y
            )
            
        # 打印结果
        # TODO 做成方法 独立出来
        if print_result == True:
            best_model_metric = pd.concat([
                self.MetricTrain.get_metric().loc[[self.best_model_name]],
                self.MetricTest.get_metric().loc[[self.best_model_name]]
            ])
            best_model_metric.index = ['Train','Test']
            #TODO 去掉前缀
            best_model_param = ', '.join([f'{param_name}={param_value}' for param_name,param_value in self.best_model_param.items()])
            message = (
                f"Best Model(CV)   : {self.best_model_name} ({self.cv_score.upper()}) \n"
                f"Hyperparameters  : {best_model_param} \n"
                f"Train Test Split : test_size={self.split_test_size}, shuffle={self.split_shuffle}, random_state=0 \n"
                f"Cross Validation : {str(self._builder.cv)} \n \n"
                f"{tabulate(best_model_metric.round(4),headers=best_model_metric.columns)} \n \n"
            )
            print(message)
        
        return self
    
    def fit_final_model(self,model='best_model',print_result=True):
        self.final_model_name = self.best_model_name if model == 'best_model' else model
        self.final_model_param = self.all_param.get(self.final_model_name)
        self.final_model = clone(self.all_model.get(self.final_model_name).best_estimator_) # 此处保留了超参数
        
        data_x = self.data.loc[:,self.col_x]
        data_y = self.data.loc[:,self.col_y].to_numpy()
        self.final_model.fit(X=data_x, y=data_y)
        self.final_model_resid = data_y - self.final_model.predict(data_x)
        
        # 最终模型在全部数据上的表现
        self.MetricFinal = self._metric(
            y_true      = self.data.loc[:,self.col_y].to_numpy(),
            y_pred      = [self.predict()],
            y_pred_name = [self.final_model_name],
            index       = None if self.col_ts is None else self.data.loc[:,self.col_ts],
            index_freq  = self.ts_freq,
            **self._metric_kwargs
        )
        if self._exp_model == True:
            self.ExpFinal = MaExplain(
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

    def predict_interval(self,new_data:pd.DataFrame=None,type:str='confidence',n_bootstrap=100,alpha=0.05) -> dict:
        self.__check_model_status(type='final')
        if new_data is None:
            new_data = self.data.loc[:,self.col_x]
        else:
            new_data = new_data.loc[:,self.col_x]
        n_row = new_data.shape[0]
        
        # 置信区间的估计
        #TODO 查阅文献改进
        if type == 'confidence':    
            sample = np.empty(shape=[n_row,n_bootstrap])
            for i in tqdm(range(n_bootstrap)):
                data = self.data.sample(frac=1,replace=True)
                mod = clone(self.final_model)
                mod.fit(X=data.loc[:,self.col_x],y=data.loc[:,self.col_y])
                sample[:,i] = mod.predict(new_data)
            low    = np.quantile(sample,q=alpha/2,axis=1)
            high   = np.quantile(sample,q=1-alpha/2,axis=1)
            result = {'down':low,'high':high}
        
        # 预测区间的估计
        elif type == 'predict':
            # 要求数据具有同方差性，因此不能处理heteroscedasticity的问题
            model = clone(self.final_model)
            new_data_pred = np.empty([n_row,n_bootstrap])
            val_resid = []
            for n in tqdm(range(n_bootstrap)):
                bs_train_idx = np.random.choice(range(n_row),size=n_row,replace=True)
                bs_valid_idx = np.array([idx for idx in range(n_row) if idx not in bs_train_idx])
                model.fit(self.data_x.loc[bs_train_idx,:],self.data_y.loc[bs_train_idx])
                val_pred = model.predict(self.data_x.loc[bs_valid_idx])
                val_resid.append(self.data_y.loc[bs_valid_idx]-val_pred)
                new_data_pred[:,n] = model.predict(new_data)
            new_data_pred = new_data_pred - new_data_pred.mean(axis=1).reshape(-1,1)
            val_resid = np.concatenate(val_resid)
                
            val_resid = np.percentile(val_resid,q=np.arange(100)) 
            train_resid = np.percentile(self.final_model_resid,q=np.arange(100))
            no_info_error = np.mean(np.abs(np.random.permutation(self.data_y) - np.random.permutation(self.predict())))
            relative_overfit_rate = np.mean(abs(val_resid.mean() - train_resid.mean()) / abs(no_info_error - train_resid.mean()))
            weight = .632 / (1 - .368 * relative_overfit_rate)
            residuals = (1-weight) * train_resid + weight * val_resid
            
            C = np.hstack([new_data_pred + o for o in residuals])  
            pred = self.predict()      
            low  = np.quantile(C,q=alpha/2,axis=1) + pred
            high = np.quantile(C,q=1-alpha/2,axis=1) + pred
            result = {'down':low,'high':high}
            
        return result
            
        
        
    def check_cv_split(self,plot=True):
        # TODO 拟合的时间、预测效果、可视化
        ...
    
    def check_novelty(self,method:str='gmm',new_data:pd.DataFrame=None,return_score:bool=False,**kwargs):
        # 当未输入new_data时，基于train_x检查test_x
        if new_data is None:
            self.__check_model_status(type='train')
            score = Novelty(train_x=self.train_x,test_x=self.test_x).get_score(method=method)
            std_resid = self.best_model_test_resid / np.std(self.best_model_test_resid)
            abs_std_resid = np.abs(std_resid)
            # train_mae = self.MetricTrain.get_metric().at[self.best_model_name,'MAE']
        
        # 当输入new_data时，基于整个数据集data检查new_data
        elif new_data is not None:
            self.__check_model_status(type='final')
            score = Novelty(train_x=self.data.loc[:,self.col_x],test_x=new_data.loc[:,self.col_x]).get_score(method=method)
            resid = new_data.loc[:,self.col_y].to_numpy() - self.predict(new_data.loc[:,self.col_x])
            abs_std_resid = np.abs(resid / np.std(resid))
            # train_mae = self.MetricFinal.get_metric().at[self.final_model_name,'MAE']
        
        if not return_score:
            data = pd.DataFrame({'score' : score, 'abs_std_residual': abs_std_resid })
            plot = corr_scatter(
                data          = data,
                x             = 'score',
                y             = 'abs_std_residual',
                smooth_method = 'qr',
                qr_quantile   = 0.95,
                # h_line        = {'Train_MAE':train_mae},
            )
            return plot
        elif return_score:
            return score
    
    def compare_train_test_metric(self):
        ...
        
    def __check_model_status(self,type='final'):
        if type == 'final':
            if self.final_model is None:
                raise Exception('模型需要先fit_final_model')
        elif type == 'train':
            if self.best_model_name is None:
                raise Exception('模型需要先fit')