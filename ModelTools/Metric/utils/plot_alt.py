import pandas as pd
import altair as alt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def plot_TvP(data,y_name,scales='fixed',select=None):
    base_Chart = alt.Chart(data)
    plot = scatter(
        base_Chart = base_Chart,
        y_name     = y_name,
        scales     = scales,
        select     = select
    )
    return plot.facet(column='Method')

def plot_Pts(data,y_name):
    select = alt.selection_interval(encodings=['x'])
    plot_v = []
    for method in data.Method.unique():
        data_method = data.loc[lambda dt:dt.Method==method]
        base_Chart = alt.Chart(data_method)
        tvp = scatter(
            base_Chart = base_Chart,
            y_name     = y_name,
            select     = select,
            scales     = 'fixed'
        )
        pts_pred = base_Chart.mark_line(size=1.5).encode(x='Time',y=alt.Y('Pred',title=y_name),color=alt.value('green'))
        pts_true = base_Chart.mark_line(size=1.5).encode(x='Time',y=alt.Y(f'True_{y_name}',title=y_name),color=alt.value('black'))
        pts = (pts_true + pts_pred ).add_selection(select).properties(width=1000,height=250)
        plot_h = alt.hconcat(tvp,pts)
        plot_v.append(plot_h)
    plot_v = alt.vconcat(*plot_v)
    return plot_v

def scatter(base_Chart,y_name,scales,select):
    # 散点
    point_black = base_Chart.mark_circle(color='black').encode(
        x = f'True_{y_name}',
        y = 'Pred',
    )
    point_gray = base_Chart.mark_circle(color='lightgray').encode(
        x = f'True_{y_name}',
        y = 'Pred',
        tooltip = alt.Tooltip(['Time:T'],format='%Y/%m/%d %H:%M:%S')
    )
    if select is not None:
        point = point_gray+point_black.transform_filter(select) 
    else:
        point = point_black
    
    # 辅助线
    line = base_Chart.mark_line(color='red',size=3).encode(
        x = f'True_{y_name}',
        y = alt.Y(f'True_{y_name}',title=f'Pred_{y_name}'),
    )
    
    plot = (point+line).properties(width=250,height=250)
    
    if scales != 'fixed':
        plot = plot.resolve_scale(x='independent',y='independent')
    
    return plot

def plot_metric_scatter(data,type):
    if type == 'bias_var':
        #TODO 增加两条辅助线
        #TODO 增加均值方差
        plot = alt.Chart(data.reset_index()).mark_circle().encode(
            x = 'resid_Median',
            y = alt.Y('resid_IQR',scale=alt.Scale(zero=False)),
            tooltip = 'index'
        )
        
    if type == 'pca':
        #PCA 分解并提取元素
        pca = Pipeline([('std',StandardScaler()),('pca',PCA())]).fit(data)
        var_ratio = pca['pca'].explained_variance_ratio_
        score = pd.DataFrame(
            pca.fit_transform(data),
            columns = [f'PC{i}' for i in range(1,data.shape[1]+1)],
            index   = data.index
        )
        loadings = pd.DataFrame(
            pca['pca'].components_,
            columns = score.columns,
            index   = data.columns
        )
        
        base_chart = alt.Chart(score.reset_index())
        point = base_chart.mark_circle().encode(
            x=alt.X('PC1',title=f'PC1 ({round(var_ratio[0]*100,2)}%)'),
            y=alt.X('PC2',title=f'PC2 ({round(var_ratio[1]*100,2)}%)'),
            tooltip='index'
        )
        vline = base_chart.mark_rule(strokeDash=[12, 6],size=1).encode(x=alt.datum(0))
        hline = base_chart.mark_rule(strokeDash=[12, 6],size=1).encode(y=alt.datum(0))
        bar_pc1 = alt.Chart(loadings.loc[:,['PC1','PC2']].reset_index()).mark_bar().encode(
            y = alt.Y('index',sort='-x'),
            x = 'PC1'
        ).properties(height=70)
        bar_pc2 = alt.Chart(loadings.loc[:,['PC1','PC2']].reset_index()).mark_bar().encode(
            x = alt.Y('index',sort='-y'),
            y = 'PC2'
        ).properties(width=70)

        plot = bar_pc1 & ((point+hline+vline)|bar_pc2)
        
    return plot