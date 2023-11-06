import altair as alt

def mts(data,var):
    base = alt.Chart(data.reset_index())
    interval_x = alt.selection_interval(encodings=['x'],zoom=False)
    
    final_plot = []
    for col in data.columns:
        final_plot.append(
            ts_main(base,col,interval_x) | ts_detail(base,col,interval_x) | scatter(base,col,var,interval_x)
        )
    final_plot = alt.vconcat(*final_plot)
    return final_plot

def ts_main(base:alt.Chart,var:str,interval_x):
    
    plot = (
        base
        .mark_line()
        .encode(
            x=alt.X('ts'),
            y=alt.Y(var)
        )
        .properties(width=800,height=200)
        .add_params(
            interval_x
        )
    )
    return plot 

def ts_detail(base:alt.Chart,var:str,interval_x):
    plot = (
        base
        .mark_line()
        .encode(
            x=alt.X('ts'),
            y=alt.Y(var,title=None)
        )
        .transform_filter(
            interval_x
        )
        .properties(width=200,height=200)
    )
    return plot 

def scatter(base:alt.Chart,var1,var2,interval_x):
    plot = (
        base 
        .mark_point(filled=True)
        .encode(
            x=alt.X(var1),
            y=alt.Y(var2).title(''),
            opacity=alt.condition(interval_x,alt.value(1),alt.value(0.01))
        )
        .properties(width=200,height=200)
    )
    return plot 

