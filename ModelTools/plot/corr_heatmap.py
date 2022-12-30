import pandas as pd
from scipy import stats
import altair as alt

from .basic.basic import BasicPlot

def corr_heatmap(
    data:pd.DataFrame,
    x:list = None,
    corr_method:str='pearson',
):
    x = data.columns if x is None else x 
    if corr_method in ['pearson','kendall','spearman']:
        corr_data = data.loc[:,x].corr(method=corr_method)
    corr_data = (
        corr_data
        .reset_index()
        .melt(id_vars='index',value_vars=x)
        .rename(columns={'index':'x1','variable':'x2','value':'corr'})
        .round(2)
    )
    basic = BasicPlot(
        data=corr_data,
        x='x1',
        y='x2'
    )
    heatmap = basic.rect(fill_color='corr')
    text = basic.text(text='corr')
    
    plot = heatmap + text
    
    return plot