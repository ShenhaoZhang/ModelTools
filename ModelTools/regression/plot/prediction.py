import pandas as pd
import plotnine as gg 

def plot_prediction(
    data    : pd.DataFrame,
    plot_var: list,
    ci_type : str,
):
    # if len(plot_var)>4:
    #     raise Exception('WRONG')
    
    aes = {'x':plot_var[0],'y':'mean'}
    if len(plot_var) >= 2:
        aes['color'] = f'factor({plot_var[1]})'
        aes['fill']  = f'factor({plot_var[1]})'
    
    plot = (
        data
        .round(2)
        .pipe(gg.ggplot)
        + gg.aes(**aes)
        + gg.geom_line()
        # + gg.geom_rug(gg.aes(x=plot_var[0]),data=self.data,inherit_aes=False)
    )
    
    # 区间估计
    ci_type = [ci_type] if not isinstance(ci_type,list) else ci_type
    for ci in ci_type:
        ci_lower = ci + '_ci_lower'
        ci_upper = ci + '_ci_upper'
        plot += gg.geom_ribbon(gg.aes(ymin=ci_lower,ymax=ci_upper),alpha=0.3,outline_type=None)
    
    # 分面
    if len(plot_var) >= 3:
        facets = f'.~{plot_var[2]}' if len(plot_var) == 3 else plot_var[2:4]
        facet_grid = gg.facet_grid(facets=facets,labeller='label_both')
        plot += facet_grid
    
    # plot += gg.geom_rug(gg.aes(x=plot_var[0]),data=self.data,inherit_aes=False)
    
    return plot