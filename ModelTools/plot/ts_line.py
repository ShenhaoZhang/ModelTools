from pandas import DataFrame
import altair as alt

from .basic.basic import BasicPlot
# from .utils.add_vh_abline import plot_add_vh_abline

def ts_line(
    data        : DataFrame,
    x           : str,
    y           : list,
    x_title     : str   = alt.Undefined,
    y_title     : str   = alt.Undefined,
    fig_width   : int   = 1000,
    fig_height  : int   = None,
    scales      : str   = 'fixed',
    color_by    : str   = alt.Undefined,
    color_legend: str   = alt.Undefined,
    add_focus   : bool  = False
):
    y = y if isinstance(y,list) else [y]
    if add_focus:
        line_width = 0.6 * fig_width
        focus_width = fig_width - line_width
    else:
        line_width = fig_width
        
    line_height = fig_height / len(x) if fig_height is not None else 200
    
    if scales == 'fixed':
        y_max = data.loc[:,y].max().max()
        y_min = data.loc[:,y].min().min()
        y_lim = [y_min,y_max]
    elif scales == 'free':
        y_lim = alt.Undefined
        
    base = BasicPlot(data=data,x=x,figure_size=[line_width,line_height])
    plot = alt.vconcat()
    selection = alt.selection_interval(encodings=['x'],empty='none')
    for y_name in y:
        base.set_attr('y',y_name)
        base.set_attr('title',y_name)
        base.set_attr('figure_size',[line_width,line_height])
        plot_line = base.line(
            y_lim        = y_lim,
            select       = selection,
            color_by     = color_by,
            color_legend = color_legend,
            y_title      = y_title,
            x_title      = x_title
        )
        
        if add_focus:
            base.set_attr('figure_size',[focus_width,line_height])
            plot_focus = base.line(
                y_lim        = alt.Undefined,
                filter       = selection,
                color_by     = color_by,
                color_legend = color_legend,
                y_title      = y_title,
                x_title      = x_title
            )
            plot_row = plot_line | plot_focus
        else:
            plot_row = plot_line
        
        plot = plot & plot_row
    
    return plot
    
