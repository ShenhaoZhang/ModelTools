from pandas import DataFrame
import altair as alt
from .basic import BasicPlot

def matrix_scatter(
    data      : DataFrame,
    x         : list  = None,
    y         : list  = None,
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
                plot_col = plot_col & plot
            plot_matrix = plot_matrix | plot_col
    else:
        for x_index,x_name in enumerate(x):
            plot_col = alt.vconcat()
            for y_index,y_name in enumerate(y):
                base = base.set_attr('x',x_name).set_attr('y',y_name)
                plot = base.scatter()
                plot_col = plot_col & plot
            plot_matrix = plot_matrix | plot_col
    
    return plot_matrix