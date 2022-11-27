from typing import Union

import numpy as np
import pandas as pd
import altair as alt 


def density(data:Union[pd.DataFrame,alt.Chart], 
            x:str,
            rotate=False,
            line_mark={},
            x_encode={},
            filter_select=None
        ):
    if isinstance(data,pd.DataFrame):
        base_chart = alt.Chart(data)
    elif isinstance(data,alt.Chart):
        base_chart = data
    
    if filter_select is not None:
        base_chart = base_chart.transform_filter(filter_select)
    
    plot = base_chart.transform_density(
        density=x,
        as_=[x,'density']
    ).encode(
        alt.X(x,type='quantitative',**x_encode),
        alt.Y('density',type='quantitative')
    ).mark_line(**line_mark)
    
    if rotate is True:
        plot = plot.encode(
            alt.X('density',type='quantitative'),
            alt.Y(x,type='quantitative',**x_encode),
            order = x 
        )
    else:
        plot = plot.encode(
            alt.X(x,type='quantitative',**x_encode),
            alt.Y('density',type='quantitative')
        )
    
    return plot 