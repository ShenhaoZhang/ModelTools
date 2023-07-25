import pandas as pd
import plotnine as gg 

def plot_grid(
    data    : pd.DataFrame,
    plot_var: list,
    ci_type : str,
    h_line  : int = None
):
    # if len(plot_var)>4:
    #     raise Exception('WRONG')
    
    aes = {'x':plot_var[0],'y':'mean'}
    if len(plot_var) >= 2:
        aes['color'] = f'factor(round({plot_var[1]},2))'
        aes['fill']  = f'factor(round({plot_var[1]},2))'
    
    plot = (
        data
        .pipe(gg.ggplot)
        + gg.aes(**aes)
        + gg.geom_line()
        # + gg.geom_rug(gg.aes(x=plot_var[0]),data=self.data,inherit_aes=False)
    )
    
    if len(plot_var) >= 2:
        plot += gg.labs(color=plot_var[1],fill=plot_var[1])
    plot += gg.labs(y='Mean')
    
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
    
    if h_line is not None:
        plot += gg.geom_hline(yintercept=h_line,linetype='--')
    
    # plot += gg.geom_rug(gg.aes(x=plot_var[0]),data=self.data,inherit_aes=False)
    
    return plot