import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
import dalex as dx

class MaExplain:
    def __init__(self,model,model_type,data_x,data_y,loss_func,quantile=None) -> None:
        self.model = model 
        self.model_type = model_type
        self.data_x = data_x
        self.data_y = data_y 
        self.loss_func = loss_func
        self.quantile = quantile
        
        warnings.filterwarnings(action='ignore',message='.*does not have valid feature names.*')
        self.Exp = dx.Explainer(
            model      = self.model,
            data       = self.data_x,
            y          = self.data_y,
            verbose    = False,
            model_type = self.model_type
        )
    
    def model_profile(self,type='ale'):
        if type == 'ale':
            plot = self.Exp.model_profile(type='ale').plot()
        elif type == 'pdp':
            plot = self.Exp.model_profile(type='pdp').plot(geom='profiles')
        return plot
    
    def model_part(self):
        def pinball_loss(y_true,y_pred):
            return metrics.mean_pinball_loss(y_true=y_true,y_pred=y_pred,alpha=self.quantile)
        total_loss_func = {
            'mse':metrics.mean_squared_error,
            'mae':metrics.mean_absolute_error,
            'mdae':metrics.median_absolute_error,
            'mape':metrics.mean_absolute_percentage_error,
            'pinball':pinball_loss
        }
        loss = total_loss_func[self.loss_func]
        
        plot = self.Exp.model_parts(loss_function=loss).plot()
        return plot
    
    def predict_part(self,new_data):
        new_data = new_data.loc[:,self.data_x.columns]
        plot = self.Exp.predict_parts(new_observation=new_data).plot()
        return plot