from typing import Union

import numpy as np
import pandas as pd
import altair as alt 


def line(data:Union[pd.DataFrame,alt.Chart], 
         x:str, 
         y:str, 
         line_mark :dict = {},
         x_encode  :dict = {},
         y_encode  :dict = {},
         domain    :list = alt.Undefined,
         chart     :dict = {},
         plot_size :list = [600,250]
    ):
    
    if isinstance(data,pd.DataFrame):
        base_chart = alt.Chart(data,**chart)
    elif isinstance(data,alt.Chart):
        base_chart = data
    
    plot = base_chart.mark_line(**line_mark).encode(
        x = alt.X(x,**x_encode),
        y = alt.Y(y,scale=alt.Scale(domain=domain),**y_encode),
    )
    plot = plot.properties(width=plot_size[0],height=plot_size[1])
    return plot


def multi_line(data:pd.DataFrame,
               x:str,
               y:Union[list,str],
               facet :str = None,
               scales:str = 'fixed',
               **kwargs
            ):
    
    if facet is not None:
        if scales == 'fixed':
            min_y = data.loc[:,y].min()
            max_y = data.loc[:,y].max()
            domain = (min_y,max_y)
            kwargs['domain']=domain
        elif scales == 'free':
            kwargs['domain']=alt.Undefined
        plot_list = []
        groups = data.loc[:,facet].unique()
        for grp in groups:
            data_grp = data.loc[data.loc[:,facet]==grp]
            plot_grp = line(data=data_grp,x=x,y=y,chart={'title':grp},**kwargs)
            plot_list.append(plot_grp)
        plot = alt.vconcat(*plot_list)
        
    elif isinstance(y,list):
        ...
    
    return plot