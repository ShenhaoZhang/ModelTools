import numpy as np
import pandas as pd 
import plotnine as gg

def plot_TVP(data:pd.DataFrame,y_name:str,error = 0.1, error_type = 'pct', type='scatter'):
        
    if type == 'scatter':
        
        if error_type=='pct':
            slope_up = 1 + error
            slope_lw = 1 - error
            inter_up,inter_lw = 0,0
            
        elif error_type == 'abs':
            slope_up,slope_lw = 1,1
            inter_up = error
            inter_lw = -error
            
        plot = (
            gg.ggplot(data=data)
            + gg.aes(x=y_name,y='pred')
            + gg.geom_point()
            + gg.geom_abline(slope=1,intercept=0,color='red')
            + gg.geom_abline(slope=slope_up,intercept=inter_up,color='red',linetype='--')
            + gg.geom_abline(slope=slope_lw,intercept=inter_lw,color='red',linetype='--')
            + gg.facet_wrap(facets='Method')
        )
    
    elif type == 'index':
        
        plot = (
            gg.ggplot(data.reset_index())
            + gg.aes(x='index')
            + gg.geom_line(gg.aes(y=y_name),color='red')
            + gg.geom_line(gg.aes(y='pred'))
            + gg.facet_wrap(facets='Method')
        )
        
    return plot


def plot_resid(data,type='index',var:pd.DataFrame=None):
    plot_data = (
        data
        .reset_index()
        .assign(abs_std_resid = lambda dt:np.abs(dt.resid/dt.resid.std())) 
    )
    
    if type == 'index':
        plot = (
            gg.ggplot(data=plot_data)
            + gg.aes(x='index',y='abs_std_resid')
            + gg.geom_line()
            + gg.facet_wrap(facets='Method')
        )
    
    elif type == 'var':
        if not isinstance(var,pd.DataFrame):
            raise Exception('WRONG')
        
        plot_data = pd.concat([plot_data,var],axis=1)
        var_col   = var.columns.to_list()
        plot = (
            plot_data
            .melt(id_vars=['abs_std_resid'],value_vars=var_col,var_name='var',value_name='value')
            .pipe(gg.ggplot)
            + gg.aes(x='value',y='abs_std_resid')
            + gg.geom_point()
            + gg.facet_wrap(facets='var',scales='free_x')
        )
    
    elif type == 'qq':
        plot = (
            gg.ggplot(data=plot_data)
            + gg.aes(sample='resid')
            + gg.geom_qq()
            + gg.geom_qq_line()
            + gg.facet_wrap(facets='Method')
        )
    
    return plot