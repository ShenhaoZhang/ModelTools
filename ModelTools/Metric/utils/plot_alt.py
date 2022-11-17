import pandas as pd
import altair as alt 

def plot_TvP(data,y_name,scales='fixed',select=None):
    base_Chart = alt.Chart(data)
    plot = scatter(
        base_Chart=base_Chart,
        y_name=y_name,
        scales=scales,
        select=select
    )
    return plot.facet(column='Method')

def plot_Pts(data,y_name):
    base_Chart = alt.Chart(data)
        
    select = alt.selection_interval(encodings=['x'])
    tvp = scatter(
        base_Chart=base_Chart,
        y_name=y_name,
        select=select,
        scales='fixed'
    )
    pts_pred = line(base_Chart=base_Chart,x = 'Time',y = 'Pred')
    pts_true = line(base_Chart=base_Chart,x = 'Time',y = f'True_{y_name}')
    pts = (pts_pred + pts_true).add_selection(select)
    return alt.hconcat(tvp,pts)

def scatter(base_Chart,y_name,scales,select):
    # 散点
    point_black = base_Chart.mark_circle(color='black').encode(
        x = f'True_{y_name}',
        y = 'Pred',
        tooltip = alt.Tooltip(['Time:T'],format='%Y/%m/%d %H:%M:%S')
    )
    point_gray = base_Chart.mark_circle(color='lightgray').encode(
        x = f'True_{y_name}',
        y = 'Pred',
        tooltip = alt.Tooltip(['Time:T'],format='%Y/%m/%d %H:%M:%S')
    )
    if select is not None:
        point = point_gray+point_black.transform_filter(select) 
    else:
        point = point_gray  
    
    # 辅助线
    line = base_Chart.mark_line(color='red',size=3).encode(
        x = f'True_{y_name}',
        y = alt.Y(f'True_{y_name}',title=f'Pred_{y_name}'),
    )
    
    plot = point+line
    if scales != 'fixed':
        plot = plot.resolve_scale(x='independent',y='independent')
    
    return plot

def line(base_Chart,x,y):
    line = base_Chart.mark_line().encode(
        x = x,
        y = y 
    )
    return line 
