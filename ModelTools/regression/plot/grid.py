import pandas as pd
import lets_plot as gg
gg.LetsPlot.setup_html()

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
    
    for var in plot_var:
        if var == plot_var[0]:
            pass
        else:
            grid_data[var] = grid_data[var].round(2)
    
    aes    = {'x':plot_var[0],'y':'mean'}
    if len(plot_var) >= 2:
        grid_data.loc[:,[plot_var[1]]] = grid_data.loc[:,[plot_var[1]]].astype('str')
        aes['color'] = plot_var[1]
        aes['fill']  = plot_var[1]
    
    plot = gg.ggplot(grid_data) + gg.aes(**aes) + gg.geom_line()
    
    if len(plot_var) >= 2:
        plot += gg.labs(color=plot_var[1],fill=plot_var[1])
    plot += gg.labs(y=y_label)
    
    # 区间估计
    plot = _add_ci_plot(plot,grid_data,ci_type)
    
    # 分面
    if len(plot_var) >= 3:
        if len(plot_var) == 3:
            facets_x     = plot_var[2]
            facets_y     = None
            facets_x_fmt = facets_x+' : {d}'
            facets_y_fmt = None
        elif len(plot_var) == 4:
            facets_x     = plot_var[2]
            facets_y     = plot_var[3]
            facets_x_fmt = facets_x+' : {d}'
            facets_y_fmt = facets_y+' : {d}'
        facet_grid = gg.facet_grid(x=facets_x,y=facets_y,x_format=facets_x_fmt,y_format=facets_y_fmt)
        plot += facet_grid
    
    if h_line is not None:
        plot += gg.geom_hline(yintercept=h_line,linetype=2,tooltips='none',color='black')
    
    # if raw_data is not None:
    #     plot += gg.geom_rug(gg.aes(x=plot_var[0]),sides='b',data=raw_data.loc[:,[plot_var[0]]],inherit_aes=False)
    
    plot += gg.theme_grey()
    
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
    aes    = {'x':'x_value','y':'mean'}
    if color_x is not None:
        grid_data.loc[:,[color_x]] = grid_data.loc[:,[color_x]].astype('str')
        aes.update({'color':color_x,'fill':color_x})
    
    if len(grid_data.x_name.unique()) <= 3:
        n_row = 1
    else:
        n_row = None
    
    plot = (
        gg.ggplot(data=grid_data)+
        gg.aes(**aes)+
        gg.facet_wrap(facets='x_name',scales=scales,nrow=n_row)+
        gg.geom_line()
    )
    
    plot = _add_ci_plot(plot,grid_data,ci_type)
    plot += gg.labs(y=y_label,x='',color=color_x,fill=color_x)
    
    if h_line is not None:
        plot += gg.geom_hline(yintercept=h_line,linetype=2,color='black')
    
    # if raw_data is not None:
    #     plot += gg.geom_rug(gg.aes(x='x_value'),sides='b',data=raw_data,inherit_aes=False)
    
    plot += gg.theme_grey()
    
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
        + gg.geom_raster()
        + gg.coord_cartesian()
        + gg.labs(fill=y_label)
    )
    
    if grid_data.loc[:,plot_var[0:2]].shape[0] <= 100:
        round_grid_data = grid_data.assign(mean=grid_data['mean'].round(1))
        plot += gg.geom_text(gg.aes(label='mean'),data=round_grid_data)
    
    # 分面
    if len(plot_var) >= 3:
        
        if len(plot_var) == 3:
            facets_x = plot_var[2]
            facets_y = None
            facets_x_fmt = facets_x+' : {d}'
            facets_y_fmt = None
        elif len(plot_var) == 4:
            facets_x = plot_var[2]
            facets_y = plot_var[3]
            facets_x_fmt = facets_x+' : {d}'
            facets_y_fmt = facets_y+' : {d}'
        
        facet_grid = gg.facet_grid(x=facets_x,y=facets_y,x_format=facets_x_fmt,y_format=facets_y_fmt)
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
        plot += gg.geom_ribbon(gg.aes(ymin=ci_lower,ymax=ci_upper),alpha=0.3,size=0.01)
    return plot

