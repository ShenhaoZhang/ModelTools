from typing import Union

import numpy as np
import pandas as pd
import altair as alt 


def scatter(data:Union[pd.DataFrame,alt.Chart], x:str, y:str, add_lm=False,add_vline=None,add_hline=None,add_other_line:list=None):
    if isinstance(data,pd.DataFrame):
        base_chart = alt.Chart(data)
    elif isinstance(data,alt.Chart):
        base_chart = data
    
    point = base_chart.mark_circle().encode(
        alt.X(x,scale = alt.Scale(zero=False),type='quantitative'),
        alt.Y(y,scale = alt.Scale(zero=False),type='quantitative'),
    )
    plot = point
    
    if add_lm is True:
        reg_line = base_chart.transform_regression(x,y).mark_line().encode(x,y)
        plot =+ reg_line
    if add_hline is not None:
        hline = base_chart.mark_rule().encode(y=alt.datum(add_hline))
        plot += hline
    if add_vline is not None:
        vline = base_chart.mark_rule().encode(x=alt.datum(add_vline))
        plot += vline
    if add_other_line:
        for line_name in add_other_line:
            plot += base_chart.mark_line().encode(x=x,y=alt.Y(line_name,type='quantitative'))
        
    return plot