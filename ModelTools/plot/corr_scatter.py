from pandas import DataFrame
from typing import Union
import altair as alt
import statsmodels.api as sm 

from .basic import BasicPlot
from .basic_reg import basic_reg

def corr_scatter(
    data           : DataFrame,
    x              : str,
    y              : str,
    fig_width      : int   = 600,
    fig_height     : int   = 400,
    lab_x          : str   = None,
    lab_y          : str   = None,
    lab_title      : str   = None,
    geom_dist      : str   = 'density',
    reg_formula    : str   = 'y~x',
    reg_method     : str   = None,
    reg_ci_level   : str   = 0.95,
    reg_pi_level   : str   = None,
    reg_qr_quantile: float = 0.5,
    stats_info     : bool  = True,
    v_line         : dict  = None,
    h_line         : dict  = None,
    v_line_pos     : str   = 'left'
):
    
    # 标签
    lab_x = x if lab_x is None else lab_x
    lab_y = y if lab_y is None else lab_y
    lab_title = alt.Undefined if lab_title is None else lab_title
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
    
    # 回归曲线及置信区间
    if isinstance(reg_qr_quantile,list) and reg_method == 'qr':
        # 多个分位数回归曲线，此时不展示区间估计
        for q in reg_qr_quantile:
            reg = basic_reg(
                data            = data,
                x               = x,
                y               = y,
                formula         = reg_formula,
                reg_method      = 'qr',
                ci_level        = None,
                pi_level        = None,
                show_stats_info = False,
                qr_quantile     = q,
            )
            scatter = scatter + reg
    elif reg_method is not None:
        # 单个回归曲线
        reg = basic_reg(
            data            = data,
            x               = x,
            y               = y,
            formula         = reg_formula,
            reg_method      = reg_method,
            ci_level        = reg_ci_level,
            pi_level        = reg_pi_level,
            show_stats_info = stats_info,
            qr_quantile     = reg_qr_quantile,
        )
        scatter = scatter + reg
    
    # 水平及垂直辅助线
    if v_line is not None:
        ...
    if h_line is not None:
        for name,value in h_line.items():
            scatter += basic.set_attr('color','black').abline(slope=0,intercept=value,name=name,name_position=v_line_pos)
    
    plot = dist_up & (scatter | dist_rt)
    plot = (
        plot
        # .properties(
        #     title={'text':lab_title,'subtitle':lab_subtitle}
        # )
        .configure_title(
            fontSize         = 20,
            baseline         = 'middle',
            subtitleFontSize = 15,
            subtitleColor    = 'grey',
            offset           = 20,
        )
    )
    
    return plot
