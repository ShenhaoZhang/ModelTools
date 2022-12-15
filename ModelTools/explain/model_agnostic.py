import warnings

import numpy as np
import pandas as pd
import dalex as dx

class MaExplain:
    def __init__(self,model,model_type,data_x,data_y) -> None:
        self.model = model 
        self.model_type = model_type
        self.data_x = data_x
        self.data_y = data_y 
        
        warnings.filterwarnings(action='ignore',message='.*does not have valid feature names.*')
        self.exp = dx.Explainer(
            model      = self.model,
            data       = self.data_x,
            y          = self.data_y,
            verbose    = False,
            model_type = self.model_type
        )
    
    def marginal_effect(self):
        plot = self.exp.model_profile(type='pdp').plot(geom='profiles')
        #TODO 用分位数代替置信区间
        return plot