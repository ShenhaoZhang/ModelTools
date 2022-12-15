import numpy as np
import pandas as pd
import altair as alt

from .basic import BasicPlot
from ..model.builder.cr_builder import CentralRegBuilder

def basic_reg(
    data            : pd.DataFrame,
    x               : str,
    y               : str,
    reg_method      : str   = 'OLS',
    reg_param       : dict  = None,
    ci_level        : float = 0.95,
    pi_level        : float = 0.95,
    int_method      : str   = 'parametric',
    show_stats_info : bool  = True
) -> alt.Chart:
    ...
    # reg_method 
    #   OLS        y = a + b * x
    #   OLS_log    y = a + b * log(x)
    #   OLS_exp    y = a + e^(b * x)
    #   OLS_pow    y = a * x^b
    #   OLS_poly   y = a + b * x + … + k * xorder
    #   OLS_spline

    # int_method = parametric / bootstrap / None
    