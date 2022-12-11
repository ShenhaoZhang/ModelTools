from pandas import DataFrame
from typing import Union

import statsmodels.api as sm 

from .basic import BasicPlot

def corr_scatter(
    data         : DataFrame,
    x            : str,
    y            : str,
    fig_width    : int   = 600,
    fig_height   : int   = 400,
    lab_x        : str   = None,
    lab_y        : str   = None,
    lab_title    : str   = None,
    geom_dist    : str   = 'density',
    smooth_method: str   = 'ols',
    stats_info   : bool  = True,
    qr_alpha     : float = 0.5,
    v_line       : dict  = None,
    h_line       : dict  = None,
    v_line_pos   : str   = 'left'
):
    
    # 标签
    lab_x = x if lab_x is None else lab_x
    lab_y = y if lab_y is None else lab_y
    lab_title = f'Relationship between {lab_x} and {lab_y}' if lab_title is None else lab_title
    lab_subtitle = []
    
    basic = BasicPlot(
        data=data,
        x=x,
        y=y 
    )
    
    # 图形大小
    scatter_width  = fig_width * 0.9
    scatter_height = fig_height * 0.9
    dist_up_width  = scatter_width
    dist_up_height = fig_height - scatter_height
    dist_rt_width  = fig_width - scatter_width
    dist_rt_height = scatter_height
    
    # 散点图
    scatter = (
        basic
        .set_attr('figure_size',[scatter_width,scatter_height])
        .scatter()
    )
    
    # 分布图
    if geom_dist == 'density':
        dist_up = (
            basic
            .set_attr('figure_size',[dist_up_width,dist_up_height])
            .density(x=x,x_title=None,y_title=None)
        )
        dist_rt = (
            basic
            .set_attr('figure_size',[dist_rt_width,dist_rt_height])
            .density(x=y,rotate=True,x_title=None,y_title=None)
        )
    elif geom_dist == 'hist':
        dist_up = (
            basic
            .set_attr('figure_size',[dist_up_width,dist_up_height])
            .hist(x=x,x_title=None,y_title=None)
        )
        dist_rt = (
            basic
            .set_attr('figure_size',[dist_rt_width,dist_rt_height])
            .hist(x=y,rotate=True,x_title=None,y_title=None)
        )
    
    # 均值回归曲线
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
    # 分位数回归曲线   
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
    
    # 水平及垂直辅助线
    if v_line is not None:
        ...
    if h_line is not None:
        for name,value in h_line.items():
            scatter += basic.abline(slope=0,intercept=value,name=name,name_position=v_line_pos)
    
    plot = dist_up & (scatter | dist_rt)
    plot = (
        plot.properties(
            title={'text':lab_title,'subtitle':lab_subtitle}
        )
        .configure_title(
            fontSize         = 20,
            baseline         = 'middle',
            subtitleFontSize = 15,
            subtitleColor    = 'grey',
            offset           = 20,
        )
    )
    
    return plot