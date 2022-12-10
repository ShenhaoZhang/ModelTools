import sys 
import os 
import warnings
from typing import Union

import pandas as pd
import altair as alt 
alt.data_transformers.disable_max_rows()

from ._base import BaseModel
from ..metric.quantile_metric import QuantileMetric
from .builder.qr_builder import QuantileRegBuilder

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::RuntimeWarning')

class QuantileRegression(BaseModel):
    def __init__(
        self, 
        data           : Union[pd.DataFrame, dict],
        col_x          : Union[str, list],
        col_y          : str,
        quantile       : float,
        col_ts         : str   = None,
        ts_freq        : str   = None,
        split_test_size: float = 0.3,
        split_shuffle  : bool  = False,
        cv_method      : str   = 'kfold',
        cv_split       : int   = 5,
        cv_shuffle     : bool  = False,
        cv_score       : str   = 'pinball',
        exp_model      : bool  = True
    ) -> None:
        super().__init__(data, col_x, col_y, col_ts, ts_freq, split_test_size, split_shuffle, cv_method, cv_split, cv_shuffle, cv_score, exp_model)
        self._init_model(quantile)
    
    def _init_model(self,quantile):
        # 配置模型的Pipeline
        #TODO 改名
        self._builder = QuantileRegBuilder(
            cv_score   = self.cv_score,
            cv_method  = self.cv_method,
            cv_shuffle = self.cv_shuffle,
            cv_split   = self.cv_split,
            quantile   = quantile
        ) 
        self._metric = QuantileMetric
        self._metric_kwargs = {'quantile':quantile}