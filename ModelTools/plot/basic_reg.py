import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
import altair as alt

from .basic import BasicPlot
from ..model.builder.cr_builder import CentralRegBuilder

def basic_reg(
    data            : pd.DataFrame,
    x:str,
    y:str,
    formula         : str   = 'y~x',
    reg_method      : str   = 'OLS',
    ci_level        : float = 0.95,
    pi_level        : float = None,
    int_method      : str   = 'parametric',
    show_stats_info : bool  = True,
    qr_quantile     : float = 0.5
) -> alt.Chart:
    ...
    # reg_method OLS / QR
    #   OLS        y = a + b * x
    #   OLS_log    y = a + b * log(x)
    #   OLS_exp    y = a + e^(b * x)
    #   OLS_pow    y = a * x^b
    #   OLS_poly   y = a + b * x + … + k * xorder
    #   OLS_spline

    # int_method = parametric / bootstrap / None
    
    formula = formula.replace('y',y).replace('x',x)
    reg_method = reg_method.upper()
    if reg_method == 'OLS':
        mod = smf.ols(formula=formula,data=data)
        mod_result = mod.fit()
    elif reg_method == 'QR':
        mod = smf.quantreg(formula=formula,data=data)
        mod_result = mod.fit(q=qr_quantile)
    else:
        raise Exception('reg_method must in ["OLS","QR"]')
    
    mod_data = mod_result.get_prediction(data).summary_frame()
    mod_data[x] = data.loc[:,x]
    
    basic_plot = BasicPlot(data=mod_data,x=x,y='mean')
    plot = basic_plot.line(y_title=y,x_title=x)
    
    if ci_level is not None or pi_level is not None:
        if pi_level is not None:
            y_up,y_down = 'obs_ci_upper','obs_ci_lower'
        else:
            y_up,y_down = 'mean_ci_upper','mean_ci_lower'
        band = basic_plot.error_band(y_up=y_up,y_down=y_down)
        plot = band + plot
    
    return plot
    
    