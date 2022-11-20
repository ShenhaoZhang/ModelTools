#%%
import numpy as np
import pandas as pd
import altair as alt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class Pca:
    def __init__(self,data:pd.DataFrame,scale=True) -> None:
        self.data = data 
        self.scale = scale
        
        if self.data.shape[1]>self.data.shape[0]:
            #TODO 会导致loadings的shape出现问题，研究此处的理论
            raise Exception(f'数据的列数{self.data.shape[1]} > 行数{self.data.shape[0]}')
        
        self.var           = None # 方差
        self.var_ratio     = None # 方差占比
        self.var_cum_ratio = None # 累计方差占比
        self.pcs           = None # 主成分重要程度（汇总方差）
        self.loadings      = None # 主成分荷载
        self.score         = None # 主成分得分
        self._fit()
    
    def _fit(self):
        # TODO 检查数据是否都是数值型
        pca = Pipeline([
            ('std',StandardScaler(with_mean=self.scale,with_std=self.scale)),
            ('pca',PCA(svd_solver='auto'))
        ])
        pca.fit(self.data)
        self._pc_name = [f'PC{i}' for i in range(1,self.data.shape[1]+1)]
        
        self.var           = pca['pca'].explained_variance_
        self.var_ratio     = pca['pca'].explained_variance_ratio_
        self.var_cum_ratio = np.cumsum(self.var_ratio).round(4)
        self.pcs = pd.DataFrame({
            'pc'           : self._pc_name,
            'var'          : self.var,
            'var_ratio'    : self.var_ratio,
            'var_cum_ratio': self.var_cum_ratio
        })
        
        self.loadings = pd.DataFrame(
            data    = pca['pca'].components_,
            columns = self._pc_name,
            index   = self.data.columns
        )
        self.score = pd.DataFrame(
            data    = pca.fit_transform(self.data),
            columns = self._pc_name,
            index   = self.data.index
        )
    
    def plot_scree(self):
        base = alt.Chart(self.pcs,title='ScreePlot')
        point = base.mark_circle(size=50).encode(
            x       = alt.X('pc'),
            y       = alt.Y('var'),
            tooltip = alt.Tooltip('var_cum_ratio')
        )
        line = base.mark_line().encode(
            x = alt.X('pc',title=None),
            y = alt.Y('var',title='Variance')
        )
        plot = (point + line).properties(width=500)
        return plot 
    
    def plot_bio(self,highlight:dict=None):
        highlight = {} if highlight is None else highlight
        data = (
            pd.concat([self.score.reset_index(),self.data.reset_index(drop=True)],axis=1)
            .assign(Highlight = lambda dt:dt.loc[:,'index'].map(highlight))
            .fillna({'Highlight':'Others'})
        )
        base = alt.Chart(
            data,
            title=f'BioPlot ({round(self.var_cum_ratio[1]*100,2)}%)'
        )
        point = base.mark_circle().encode(
            x       = alt.X('PC1',title=f'PC1 ({round(self.var_ratio[0]*100,2)}%)'),
            y       = alt.X('PC2',title=f'PC2 ({round(self.var_ratio[1]*100,2)}%)'),
            tooltip = alt.Tooltip(['index']+self.data.columns.to_list())
        )
        if not (data.Highlight=='Others').all():
            point = point.encode(
                color = alt.Color('Highlight:N',title=None,
                                  legend=alt.Legend(direction='horizontal',orient='bottom',titleAnchor='middle'))
                )
        
        vline = base.mark_rule(strokeDash=[12, 6],size=1).encode(x=alt.datum(0))
        hline = base.mark_rule(strokeDash=[12, 6],size=1).encode(y=alt.datum(0))
        
        bar_width = 12 * len(self._pc_name) # 根据指标的数量动态调整条形图的大小
        bar_pc1 = alt.Chart(self.loadings.loc[:,['PC1','PC2']].reset_index()).mark_bar().encode(
            y = alt.Y('index',sort='-x',title=None),
            x = 'PC1'
        ).properties(height=bar_width)
        bar_pc2 = alt.Chart(self.loadings.loc[:,['PC1','PC2']].reset_index()).mark_bar().encode(
            x = alt.Y('index',sort='-y',title=None),
            y = 'PC2'
        ).properties(width=bar_width)

        plot = bar_pc1 & ((point+hline+vline)|bar_pc2)
        return plot 

#%%
if __name__ == '__main__':
    #%%
    from sklearn.datasets import make_spd_matrix
    
    n = 5
    rng = np.random.default_rng(0)
    x = rng.multivariate_normal(
        mean=rng.normal(size=n,scale=2),
        cov=make_spd_matrix(n_dim=n,random_state=0),
        size=5
    )
    df = pd.DataFrame(data=x,columns=[f'x{i}' for i in range(n)])
    pc = Pca(df)
    pc.plot_scree()





# %%
