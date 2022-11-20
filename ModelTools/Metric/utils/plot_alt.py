import pandas as pd
import altair as alt 

def plot_TvP(data,y_name,scales='fixed',select=None):
    base_Chart = alt.Chart(data)
    plot = scatter(
        base_Chart = base_Chart,
        y_name     = y_name,
        scales     = scales,
        select     = select
    )
    return plot.facet(column='Method')

def plot_Pts(data,y_name):
    select = alt.selection_interval(encodings=['x'])
    plot_v = []
    for method in data.Method.unique():
        data_method = data.loc[lambda dt:dt.Method==method]
        base_Chart = alt.Chart(data_method)
        tvp = scatter(
            base_Chart = base_Chart,
            y_name     = y_name,
            select     = select,
            scales     = 'fixed'
        )
        pts_pred = base_Chart.mark_line(size=1.5).encode(x='Time',y=alt.Y('Pred',title=y_name),color=alt.value('green'))
        pts_true = base_Chart.mark_line(size=1.5).encode(x='Time',y=alt.Y(f'True_{y_name}',title=y_name),color=alt.value('black'))
        pts = (pts_true + pts_pred ).add_selection(select).properties(width=1000,height=250)
        plot_h = alt.hconcat(tvp,pts)
        plot_v.append(plot_h)
    plot_v = alt.vconcat(*plot_v)
    return plot_v

def scatter(base_Chart,y_name,scales,select):
    # 散点
    point_black = base_Chart.mark_circle(color='black').encode(
        x = f'True_{y_name}',
        y = 'Pred',
    )
    point_gray = base_Chart.mark_circle(color='lightgray').encode(
        x = f'True_{y_name}',
        y = 'Pred',
        tooltip = alt.Tooltip(['Time:T'],format='%Y/%m/%d %H:%M:%S')
    )
    if select is not None:
        point = point_gray+point_black.transform_filter(select) 
    else:
        point = point_black
    
    # 辅助线
    line = base_Chart.mark_line(color='red',size=3).encode(
        x = f'True_{y_name}',
        y = alt.Y(f'True_{y_name}',title=f'Pred_{y_name}'),
    )
    
    plot = (point+line).properties(width=250,height=250)
    
    if scales != 'fixed':
        plot = plot.resolve_scale(x='independent',y='independent')
    
    return plot

def plot_metric_bias_var(data):
    #TODO 增加两条辅助线 总体水平线
    #TODO 增加x轴的零点线
    #TODO 增加均值方差
    plot = alt.Chart(data.reset_index()).mark_circle().encode(
        x = 'resid_Median',
        y = alt.Y('resid_IQR',scale=alt.Scale(zero=False)),
        tooltip = 'index'
    )
    if not (data.Highlight=='Others').all():
        plot = plot.encode(color = alt.Color('Highlight:N',title=None))
    return plot