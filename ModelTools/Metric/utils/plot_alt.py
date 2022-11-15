import altair as alt 

def plot_TvP(data,y_name,scales='fixed'):
    base = alt.Chart(data)
    point = base.mark_circle().encode(
        x = f'True_{y_name}',
        y = 'Pred'
    )
    line = base.mark_line(color='red',size=2).encode(
        x = f'True_{y_name}',
        y = alt.Y(f'True_{y_name}',title=f'Pred_{y_name}'),
    )
    
    plot = (point+line).facet('method')
    if scales != 'fixed':
        plot = plot.resolve_scale(x='independent',y='independent')
    
    return plot