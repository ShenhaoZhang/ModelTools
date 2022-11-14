import itertools

import numpy as np
import pandas as pd 
import plotnine as gg

class DataExplorer:
    
#TODO 数据清洗 探索性可视化
    
    def __init__(self,data:pd.DataFrame) -> None:
        self.data = data 
    
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