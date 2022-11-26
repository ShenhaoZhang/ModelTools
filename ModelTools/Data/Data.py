import itertools

import numpy as np
import pandas as pd 
import plotnine as gg
import altair as alt

class Data:
    
#TODO 数据清洗 探索性可视化
#TODO x和y的时序图+xy的散点图
    
    def __init__(self,data:pd.DataFrame,col_x:list=None,col_y:str=None) -> None:
        self.data = data 
        self.col_x = col_x
        self.col_y = col_y 
    
    def plot_ts_and_scatter(self,col_x:str,col_y:str=None):
        if col_y is None:
            col_y = self.col_y 
        
        select = alt.selection_interval(encodings=['x'])
        base_chart = alt.Chart(self.data.assign(ts_index=lambda dt:np.arange(len(dt))))
        
        point = base_chart.mark_circle(color='lightgray').encode(
            x = alt.X(col_x,type='quantitative'),
            y = alt.Y(col_y,type='quantitative'),
        ).properties(height=400,width=400)
        point = point + point.mark_circle(color='black').transform_filter(select)
        
        top_hist_base = base_chart.encode(
            x = alt.X(col_x,type='quantitative'),
            y = alt.Y('density',type='quantitative')
        ).properties(height=100,width=400)
        top_hist = top_hist_base.mark_line(color='lightgray').transform_density(
            density=col_x,
            as_=[col_x,'density']
        ) + top_hist_base.mark_line(color='black').transform_filter(select).transform_density(
            density=col_x,
            as_=[col_x,'density']
        )
        
        right_hist_base = base_chart.encode(
            y = alt.Y(col_y,type='quantitative'),
            x = alt.X('density',type='quantitative'),
            order = col_y
        ).properties(height=400,width=100)
        right_hist = right_hist_base.mark_line(color='light').transform_density(
            density=col_y,
            as_=[col_y,'density']
        ) + right_hist_base.mark_line(color='black').transform_filter(select).transform_density(
            density=col_y,
            as_=[col_y,'density']
        )
        
        up_line = base_chart.mark_line().encode(
            x = alt.X('ts_index'),
            y = alt.Y(col_x),
        ).properties(height=250,width=600).add_selection(select)
        dw_line = base_chart.mark_line().encode(
            x = alt.X('ts_index'),
            y = alt.Y(col_y)
        ).properties(height=250,width=600).add_selection(select)
        
        plot_left = up_line & dw_line
        plot_right = top_hist & (point|right_hist)
        plot = alt.hconcat(plot_left,plot_right)
                
        return plot
        # return point
        
        
    def plot_scatter_matrix(self):
        pdf = []
        for a1, b1 in itertools.combinations(self.data.columns, 2):
            for (a,b) in ((a1, b1), (b1, a1)):
                sub = self.data[[a, b]].rename(columns={a: "x", b: "y"}).assign(a=a, b=b)
                pdf.append(sub)

        plot = gg.ggplot(pd.concat(pdf))
        plot += gg.geom_point(gg.aes('x','y'))
        plot += gg.facet_grid('b~a', scales='free')
        return plot
    
if __name__ == '__main__':
    from plotnine.data import mtcars
    de = DataExplorer(data=mtcars.select_dtypes(include='number').iloc[:,1:4])
    print(de.plot_scatter_matrix())