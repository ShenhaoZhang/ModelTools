from pandas import DataFrame
import altair as alt
from ..basic.basic import BasicPlot

def plot_add_vh_abline(
    plot,
    basic_plot  : BasicPlot,
    v_line      : dict  = None,
    h_line      : dict  = None,
    v_storkeDash: str   = alt.Undefined,
    h_storkeDash: str   = alt.Undefined,
    v_line_color: str   = 'black',
    h_line_color: str   = 'black'
):
    basic = basic_plot
    basic_origin_color = basic.color
    if v_line is not None:
        for name,value in v_line.items():
            plot += basic.set_attr('color',v_line_color).abline(
                slope      = None,
                intercept  = value,
                name       = name,
                storkeDash = v_storkeDash
                # name_position = v_line_name_posit
            )
    elif h_line is not None:
        for name,value in h_line.items():
            plot += basic.set_attr('color',h_line_color).abline(
                slope      = 0,
                intercept  = value,
                storkeDash = h_storkeDash
                # name_position = h_line_name_posit
            )
    basic.set_attr('color',basic_origin_color)
    return plot