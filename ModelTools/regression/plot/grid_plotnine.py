import pandas as pd
import plotnine as gg 

def plot_grid_1d(
    grid_data: pd.DataFrame,
    raw_data : pd.DataFrame,
    plot_var : list,
    ci_type  : str,
    h_line   : int = None,
    y_label  : str = 'y',
):
    if len(plot_var)>4:
        raise Exception('WRONG')
    
    aes = {'x':plot_var[0],'y':'mean'}
    if len(plot_var) >= 2:
        aes['color'] = f'factor(round({plot_var[1]},2))'
        aes['fill']  = f'factor(round({plot_var[1]},2))'
    
    plot = gg.ggplot(grid_data) + gg.aes(**aes) + gg.geom_line()
    
    if len(plot_var) >= 2:
        plot += gg.labs(color=plot_var[1],fill=plot_var[1])
    plot += gg.labs(y=y_label)
    
    # 区间估计
    plot = _add_ci_plot(plot,grid_data,ci_type)
    
    # 分面
    if len(plot_var) >= 3:
        facets = f'.~{plot_var[2]}' if len(plot_var) == 3 else plot_var[2:4]
        facet_grid = gg.facet_grid(facets=facets,labeller='label_both')
        plot += facet_grid
    
    if h_line is not None:
        plot += gg.geom_hline(yintercept=h_line,linetype='--')
    
    if raw_data is not None:
        plot += gg.geom_rug(gg.aes(x=plot_var[0]),sides='b',data=raw_data.loc[:,[plot_var[0]]],inherit_aes=False)
    
    return plot

def plot_all_grid_1d(
    grid_data: pd.DataFrame,
    raw_data : pd.DataFrame,
    free_y   : bool,
    ci_type  : str,
    y_label  : str,
    color_x  : str,
    h_line   : int = None,
):
    
    scales = 'free' if free_y else 'free_x'
    aes = {'x':'x_value','y':'mean'}
    if color_x is not None:
        aes.update({
            'color':f'factor(round({color_x},4))',
            'fill' :f'factor(round({color_x},4))'
        })
    
    plot = (
        gg.ggplot(data=grid_data)+
        gg.aes(**aes)+
        gg.facet_wrap(facets='x_name',scales=scales)+
        gg.geom_line()
    )
    
    plot = _add_ci_plot(plot,grid_data,ci_type)
    plot += gg.labs(y=y_label,x='',color=color_x,fill=color_x)
    
    if h_line is not None:
        plot += gg.geom_hline(yintercept=h_line,linetype='--')
    
    if raw_data is not None:
        plot += gg.geom_rug(gg.aes(x='x_value'),sides='b',data=raw_data,inherit_aes=False)
    
    return plot

def plot_grid_2d(
    grid_data: pd.DataFrame,
    plot_var : list,
    y_label  : str = 'y',
):
    if len(plot_var)>4:
        raise Exception('WRONG')
    
    aes = {'x':plot_var[0],'y':plot_var[1],'fill':'mean'}
    
    plot = (
        gg.ggplot(grid_data) 
        + gg.aes(**aes) 
        + gg.geom_tile()
        + gg.coord_cartesian(expand=False)
        + gg.labs(fill=y_label)
    )
    
    if grid_data.loc[:,plot_var[0:2]].shape[0] <= 100:
        plot += gg.geom_text(gg.aes(label='round(mean,1)'))
    
    # 分面
    if len(plot_var) >= 3:
        facets = f'.~{plot_var[2]}' if len(plot_var) == 3 else plot_var[2:4]
        facet_grid = gg.facet_grid(facets=facets,labeller='label_both')
        plot += facet_grid
        
    return plot

def _add_ci_plot(plot,data,ci_type):
    # 区间估计
    ci_type = [ci_type] if not isinstance(ci_type,list) else ci_type
    for ci in ci_type:
        ci_lower = ci + '_ci_lower'
        ci_upper = ci + '_ci_upper'
        if ci_lower not in data.columns:
            continue
        plot += gg.geom_ribbon(gg.aes(ymin=ci_lower,ymax=ci_upper),alpha=0.3,outline_type=None)
    return plot

