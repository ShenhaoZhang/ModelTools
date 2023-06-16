import pandas as pd
import plotnine as gg

def plot_distribution(
    data:pd.DataFrame,
    var:list=None,
    facets=True,
    scales='free',
):
    var = data.columns if var is None else var
    data = (
        data
        .melt(value_vars=var)
    )
    
    plot = (
        gg.ggplot(data)
        + gg.aes(x='value')
        + gg.geom_density()
    )
    
    if facets == True:
        plot += gg.facet_wrap(facets='variable',scales=scales)
    else:
        plot += gg.aes(color='variable')
        
    return plot