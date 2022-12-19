from typing import Union
from pandas import DataFrame
import altair as alt
from .basic.basic import BasicPlot
from .utils.scales import get_scales

def matrix_scatter(
    data      : DataFrame,
    x         : list  = None,
    y         : list  = None,
    diag_line : bool  = False,
    fig_width : int   = 600,
    fig_height: int   = 600,
):
    # TODO 增加分类数据类型
    x = data.columns if x is None else x 
    y = x if y is None else y 
    
    plot_width = fig_width / len(y)
    plot_height = fig_height / len(x)
    base = BasicPlot(data=data,figure_size=[plot_width,plot_height])
    
    plot_matrix = alt.hconcat()
    if set(x) == set(y):
        for x_index,x_name in enumerate(x):
            plot_col = alt.vconcat()
            for y_index,y_name in enumerate(y):
                base = base.set_attr('x',x_name).set_attr('y',y_name)
                if x_index == y_index:
                    plot = base.density()
                elif x_index > y_index:
                    plot = base.scatter()
                elif x_index < y_index:
                    plot = base.scatter()
                if diag_line:
                    plot += plot_diag_line(base)
                plot_col = plot_col & plot
            plot_matrix = plot_matrix | plot_col
    else:
        for x_index,x_name in enumerate(x):
            plot_col = alt.vconcat()
            for y_index,y_name in enumerate(y):
                base = base.set_attr('x',x_name).set_attr('y',y_name)
                plot = base.scatter()
                if diag_line:
                    plot += plot_diag_line(base)
                plot_col = plot_col & plot
            plot_matrix = plot_matrix | plot_col
    
    return plot_matrix

def matrix_scatter_wrap(
    data      : DataFrame,
    x         : Union[list,str]  = None,
    y         : Union[list,str]  = None,
    scales    : str = 'fixed',
    diag_line : bool  = False,
    n_col:int = 4,
    fig_width : int   = 600,
    fig_height: int   = 600,
):
    x = [x] if isinstance(x,str) else x 
    y = [y] if isinstance(x,str) else y 
    if len(x)>1 and len(y)==1:
        y = y * len(x)
    if len(y)>1 and len(x)==1:
        x = x * len(y)
    
    base = BasicPlot(data=data).set_attr('figure_size',(300,300))
    plot_matrix = alt.vconcat()
    plot_row = alt.hconcat()
    x_lim,y_lim = get_scales(data,x,y,scales)
    for idx in range(len(x)):
        x_i = x[idx]
        y_i = y[idx]
        plot = base.set_attr('x',x_i).set_attr('y',y_i).scatter(
            x_lim=x_lim,
            y_lim=y_lim
        )
        if diag_line:
            plot = plot + plot_diag_line(base,x_lim,y_lim)
        plot_row = plot_row | plot 
        if (idx+1) % n_col == 0:
            plot_matrix = plot_matrix & plot_row
            plot_row = alt.hconcat()
    plot = plot_row if idx+1<n_col else plot_matrix
    return plot
    
    
        
    

def plot_diag_line(base,x_lim=alt.Undefined,y_lim=alt.Undefined):
    origin_color = base.color
    plot = base.set_attr('color','red').abline(slope=1,intercept=0,x_lim=x_lim,y_lim=y_lim)
    base.set_attr('color',origin_color)
    return plot