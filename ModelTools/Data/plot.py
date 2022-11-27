import numpy as np
import pandas as pd
import altair as alt 

def ts_scatter(data:pd.DataFrame, x:str, y:str, ts:str = None, add_lm=False) -> alt.HConcatChart:
    """
    两个变量的时间序列图及散点图

    Parameters
    ----------
    data : pd.DataFrame
        数据
    x : str
        散点图的x轴对应变量
    y : str
        散点图的y轴对应变量
    ts : str, optional
        时序的x轴对应变量, 当设为None时, 用连续的序列替代, by default None
    add_lm : bool, optional
        图形是否增加OLS线性回归曲线, by default False

    Returns
    -------
    alt.HConcatChart
        组合图形
    """
    if ts is None:
        data = data.assign(_ts_index=lambda dt:np.arange(len(dt)))
        ts = '_ts_index'
    base_chart = alt.Chart(data)
    select = alt.selection_interval(encodings=['x'])
    
    # 散点图
    point_back = base_chart.mark_circle(color='lightgray').encode(
        x = alt.X(x,type='quantitative'),
        y = alt.Y(y,type='quantitative'),
    ).properties(height=400,width=400)
    point_highlight = point_back.mark_circle(color='black').transform_filter(select)
    point = point_back + point_highlight 
    if add_lm == True:
        line_regression_back = point_back.transform_regression(x,y).mark_line(color='lightgray')
        line_regression_highlight = point_back.transform_filter(select).transform_regression(x,y).mark_line(color='black')
        point += line_regression_back + line_regression_highlight
    
    # 密度曲线（上）
    top_hist_base = base_chart.encode(
        x = alt.X(x,type='quantitative',title=None),
        y = alt.Y('density',type='quantitative')
    ).properties(height=100,width=400)
    top_hist = top_hist_base.mark_line(color='lightgray').transform_density(
        density=x,
        as_=[x,'density']
    ) + top_hist_base.mark_line(color='black').transform_filter(select).transform_density(
        density=x,
        as_=[x,'density']
    )
    
    # 密度曲线（右）
    right_hist_base = base_chart.encode(
        y = alt.Y(y,type='quantitative',title=None),
        x = alt.X('density',type='quantitative'),
        order = y
    ).properties(height=400,width=100)
    right_hist = right_hist_base.mark_line(color='light').transform_density(
        density=y,
        as_=[y,'density']
    ) + right_hist_base.mark_line(color='black').transform_filter(select).transform_density(
        density=y,
        as_=[y,'density']
    )
    
    # 时序图（上）
    up_line = base_chart.mark_line().encode(
        x = alt.X(ts,title=None),
        y = alt.Y(x),
    ).properties(height=250,width=600).add_selection(select)
    
    # 时序图（下）
    dw_line = base_chart.mark_line().encode(
        x = alt.X(ts,title='Time'),
        y = alt.Y(y)
    ).properties(height=250,width=600).add_selection(select)
    
    # 图形组合
    plot_left = up_line & dw_line
    plot_right = top_hist & (point|right_hist)
    plot = alt.hconcat(plot_left,plot_right)
            
    return plot