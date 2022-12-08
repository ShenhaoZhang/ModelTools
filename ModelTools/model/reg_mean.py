
#TODO 增加训练和测试上模型的效果比较
#TODO 如果工况分布不均匀 且 成块出现，不应split_shuffle为False，这样交叉验证的结果必然不好，应该分组抽样，cv的shuffle也应该分组抽样多特征如何分组？首先PCA？
#TODO 预测区间和置信区间的可视化

import sys 
import os 
import warnings
from typing import Union

import pandas as pd
import altair as alt 
alt.data_transformers.disable_max_rows()

from ._base import BaseModel
from .builder.rm_builder import MeanRegBuilder
from ..metric.central_metric import CentralMetric

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::RuntimeWarning')

class MeanRegression(BaseModel):
    def __init__(
        self, 
        data           : Union[pd.DataFrame, dict],
        col_x          : Union[str, list],
        col_y          : str,
        col_ts         : str   = None,
        ts_freq        : str   = None,
        split_test_size: float = 0.3,
        split_shuffle  : bool  = False,
        cv_method      : str   = 'kfold',
        cv_split       : int   = 5,
        cv_shuffle     : bool  = False,
        cv_score       : str   = 'mse',
        exp_model      : bool  = True
    ) -> None:
        super().__init__(data, col_x, col_y, col_ts, ts_freq, split_test_size, split_shuffle, cv_method, cv_split, cv_shuffle, cv_score, exp_model)
        self._init_model(MeanRegBuilder,CentralMetric)
    