from typing import Union

from .basic import BasicPlot

def ts2_scatter(
    data,
    ts,
    x,
    y,
    fig_width = 900,
    fig_height = None,
    lab_x = None,
    lab_y = None,
    lab_title=None,
    lab_subtitle=None,
    geom_dist = 'density'
):
    base = BasicPlot(
        data=data,
        x=x,
        y=y
    )
    
    if fig_width is not None:
        fig_height = fig_width / 3
    elif fig_height is not None:
        fig_width = fig_height * 3
    else:
        print('建议fig_width/fig_height=3')
    
    scatter_width  = fig_width * 0.3
    scatter_height = fig_height * 0.9
    dist_up_width  = scatter_width
    dist_up_height = fig_height - scatter_height
    dist_rt_width  = dist_up_height
    dist_rt_height = dist_up_width
    line_up_width  = fig_width - scatter_width - dist_rt_width
    line_up_height = fig_height * 0.5
    line_dw_width  = line_up_width
    line_dw_height = line_up_height
    
    # 散点图
    scatter = base.set_attr('figure_size',[scatter_width,scatter_height]).scatter()
    
    # 分布图
    if geom_dist == 'density':
        dist_up = (
            base
            .set_attr('figure_size',[dist_up_width,dist_up_height])
            .density(x='x',x_title=None,y_title=None)
        )
        dist_rt = (
            base
            .set_attr('figure_size',[dist_rt_width,dist_rt_height])
            .density(x='y',rotate=True,x_title=None,y_title=None)
        )
    elif geom_dist == 'hist':
        dist_up = (
            base
            .set_attr('figure_size',[dist_up_width,dist_up_height])
            .hist(x='x',x_title=None,y_title=None)
        )
        dist_rt = (
            base
            .set_attr('figure_size',[dist_rt_width,dist_rt_height])
            .hist(x='y',rotate=True,x_title=None,y_title=None)
        )
    
    # 线图
    line_up = (
        base
        .set_attr('figure_size',[line_up_width,line_up_height])
        .set_attr(x,ts)
        .set_attr(y,y)
        .line(x_title=None)
    )
    line_dw = (
        base
        .set_attr('figure_size',[line_dw_width,line_dw_height])
        .set_attr(x,ts)
        .set_attr(y,x)
        .line()
    )
    
    plot = (line_up & line_dw) | (dist_up & (scatter|dist_rt))
    
    return plot