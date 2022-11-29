import numpy as np
import pandas as pd
import altair as alt 

from .corr import scatter
from .evo import line
from .dist import density

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
    point_back = scatter(data=base_chart,x=x,y=y,add_lm=add_lm,point_mark={'color':'lightgray'},lm_mark={'color':'lightgray'})
    point_highlight = scatter(data=base_chart,x=x,y=y,add_lm=add_lm,point_mark={'color':'black'},lm_mark={'color':'black'}).transform_filter(select)
    point = point_back + point_highlight
    point = point.properties(height=400,width=400)
    
    # 密度曲线（上和右）
    top_hist_back = density(data=base_chart,x=x,line_mark={'color':'lightgray'},x_encode={'title':None})
    top_hist_highlight = density(data=base_chart,x=x,line_mark={'color':'black'},x_encode={'title':None},
                                 filter_select=select)
    top_hist = top_hist_back + top_hist_highlight
    top_hist = top_hist.properties(height=100,width=400)
    
    right_hist_back = density(data=base_chart,x=y,line_mark={'color':'lightgray'},x_encode={'title':None},rotate=True)
    right_hist_highlight = density(data=base_chart,x=y,line_mark={'color':'black'},x_encode={'title':None},filter_select=select,rotate=True)
    right_hist = right_hist_back + right_hist_highlight
    right_hist = right_hist.properties(height=400,width=100)
    
    # 时序图（上和下）
    up_line = line(data=base_chart,x=ts,y=x,x_encode={'title':None}).properties(height=250,width=600).add_selection(select)
    dw_line = line(data=base_chart,x=ts,y=y,x_encode={'title':'Time'}).properties(height=250,width=600).add_selection(select)
    
    # 图形组合
    plot_left = up_line & dw_line
    plot_right = top_hist & (point|right_hist)
    plot = alt.hconcat(plot_left,plot_right)
            
    return plot

def multi_line_facet(data:pd.DataFrame):
    ...