from typing import Union

import pandas as pd
import statsmodels.api as sm 
import altair as alt

from .basic import BasicPlot

def corr_scatter(
    data,
    x,
    y,
    fig_width = 600,
    fig_height = 400,
    lab_x = None,
    lab_y = None,
    lab_title=None,
    lab_subtitle=None,
    smooth_method = 'ols',
    stats_info = True,
    qr_alpha=0.5,
    v_line:dict = None,
    h_line:dict = None,
    v_line_pos = 'left'
):
    
    lab_x = x if lab_x is None else lab_x
    lab_y = y if lab_y is None else lab_y
    lab_title = f'Relationship between {lab_x} and {lab_y}' if lab_title is None else lab_title
    lab_subtitle = []
    
    basic = BasicPlot(
        data=data,
        x=x,
        y=y 
    )
    scatter_width  = fig_width * 0.9
    scatter_height = fig_height * 0.9
    hist_up_width  = scatter_width
    hist_up_height = fig_height - scatter_height
    hist_rt_width  = fig_width - scatter_width
    hist_rt_height = scatter_height
    
    scatter = (
        basic
        .set_attr('figure_size',[scatter_width,scatter_height])
        .scatter()
    )
    hist_up = (
        basic
        .set_attr('figure_size',[hist_up_width,hist_up_height])
        .hist(x=x)
    )
    hist_rt = (
        basic
        .set_attr('figure_size',[hist_rt_width,hist_rt_height])
        .hist(x=y,rotate=True)
    )
    
    if smooth_method == 'ols':
        smooth = basic.smooth(method='linear')
        scatter += smooth
        
        X = data.loc[:,x]
        X = sm.add_constant(X)
        Y = data.loc[:,y]
        ols = sm.OLS(Y,X).fit()
        
        if stats_info:
            info_mod = (
                f'Regressiond Method = Linear Regression,  '
                f'Sample = {int(ols.nobs)},  '
                f'R2 = {round(ols.rsquared,2)},  '
                f'adj_R2 = {round(ols.rsquared_adj,2)},  '
                f'F_pvalue = {round(ols.f_pvalue,4)}'
            )
            lab_subtitle.append(info_mod)
            for param in ols.params.index:
                param_name = 'Intercept' if param == 'const' else param
                info_coef = (
                    f'{param_name} :  '
                    f'coef = {round(ols.params[param],2)},  '
                    f't = {round(ols.tvalues[param],2)},  '
                    f'p = {round(ols.pvalues[param],4)},  '
                    f'CI_0.95 = [ {round(ols.conf_int().at[param,0],2)}, {round(ols.conf_int().at[param,1],2)} ]'
                )
                lab_subtitle.append(info_coef)
                
    elif smooth_method == 'qr':
        X = data.loc[:,x]
        X = sm.add_constant(X)
        Y = data.loc[:,y]
        qr = sm.QuantReg(Y,X).fit(q=qr_alpha)
        scatter += basic.abline(slope=qr.params[x],intercept=qr.params['const'])
        
        if stats_info:
            info_mod = (
                f'Regressiond Method = Quantile Regression(alpha={qr_alpha}),  '
                f'Sample = {int(qr.nobs)},  '
                f'Pseudo R2 = {round(qr.prsquared,2)}  '
            )
            lab_subtitle.append(info_mod)
            for param in qr.params.index:
                param_name = 'Intercept' if param == 'const' else param
                info_coef = (
                    f'{param_name} :  '
                    f'coef = {round(qr.params[param],2)},  '
                    f't = {round(qr.tvalues[param],2)},  '
                    f'p = {round(qr.pvalues[param],4)},  '
                    f'CI_0.95 = [ {round(qr.conf_int().at[param,0],2)}, {round(qr.conf_int().at[param,1],2)} ]'
                )
                lab_subtitle.append(info_coef)
    
    if v_line is not None:
        ...
    if h_line is not None:
        for name,value in h_line.items():
            scatter += basic.abline(slope=0,intercept=value,name=name,name_position=v_line_pos)
    
    plot = hist_up & (scatter | hist_rt)
    
    plot = (
        plot.properties(
            title={'text':lab_title,'subtitle':lab_subtitle}
        )
        .configure_title(
            fontSize=20,
            baseline='middle',
            subtitleFontSize=15,
            subtitleColor='grey',
            offset=20,
        )
    )
    
    return plot