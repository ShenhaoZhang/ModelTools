import numpy as np
import pandas as pd
import plotnine as gg
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

def gg_Tvp(data,y_name,y_pred_name,caption,add_lm=False,add_outlier=False,add_quantile=False,scales='fixed',figure_size=(10, 5)):
    gg.options.figure_size = figure_size
    plot = (
        gg.ggplot(data)
        + gg.aes(x=f'True_{y_name}',y='Pred')
        + gg.geom_point()
        + gg.facet_wrap(facets='Method',scales=scales)
        + gg.labs(
            title=f'True_{y_name} VS Predict_{y_name}',
            caption=caption,
            x = f'True_{y_name}',
            y = f'Pred_{y_name}'
        )
    )
    plot = plot + gg.theme(subplots_adjust={'wspace': 0.25,'hspace': 0.25}) if scales!='fixed' else plot
    
    if add_lm:
        plot += gg.geom_smooth(Method='lm',color='blue')
        
    if add_outlier:
        plot = plot + gg.geom_point(gg.aes(color='Outlier')) + gg.scale_color_manual(values=['black','red'])
        
    if add_quantile:
        for q in [0.05,0.95]:
            mod = Pipeline([
                ('std',StandardScaler()),
                ('spline',PolynomialFeatures()),
                ('qr',QuantileRegressor(quantile=q,alpha=0,solver='highs')),
            ])
            x_sample, y_hat, method_sample = np.array([]), np.array([]) ,np.array([])
            for name in y_pred_name:
                data_name = data.loc[lambda dt:dt.Method==name,:]
                x = data_name.loc[:,f'True_{y_name}'].to_numpy()
                y = data_name.loc[:,'Pred'].to_numpy()
                mod.fit(x.reshape(-1,1),y)
                y_hat = np.append(y_hat,mod.predict(X=x.reshape(-1,1)))
                x_sample = np.append(x_sample,x)
                method_sample = np.append(method_sample,[name]*len(x))
                
            plot += gg.geom_line(
                data=pd.DataFrame({'x':x_sample,'y':y_hat,'Method':method_sample}),
                mapping = gg.aes(x='x',y='y'),
                color='red',
                linetype='--'
            )
    plot += gg.geom_abline(slope=1,intercept=0,color='red',size=1)
            
    return plot


def gg_Pts(data,y_name,caption,time_limit=None,drop_anomaly=False,figure_size=(10, 5),scales='fixed'):
    gg.options.figure_size = figure_size
    #TODO 当预测值或实际值存在较大的异常值时，图形会缩放，导致难以观察，调整
    if drop_anomaly is True:
        # plot_data = plot_data.loc[lambda dt:dt.]
        pass
    else:
        plot_data = data
        
    plot = (
        gg.ggplot(plot_data)
        + gg.aes(x = 'Time')
        + gg.geom_line(gg.aes(y=f'True_{y_name}',color="'True'"))
        + gg.geom_line(gg.aes(y='Pred',color="'Pred'"))
        + gg.facet_wrap(facets='Method',ncol=1,scales=scales)
        + gg.scale_color_manual(values=['black','green'])
        + gg.labs(
            color = ' ',
            title = f'Time Series for True_{y_name} and Predict_{y_name}',
            caption = caption,
            y = y_name
        )
    )
    
    if time_limit:
        plot += gg.scale_x_continuous(limits=time_limit)
        #TODO 增加时间index的处理
    
    return plot

def gg_Rts(data,caption,iqr,add_iqr_line=False,time_limit=None,figure_size=(10, 5),scales='fixed'):
    gg.options.figure_size = figure_size
    plot = (
        gg.ggplot(data)
        + gg.geom_line(gg.aes(x='Time',y='Resid'))
        + gg.geom_hline(yintercept=0,size=1,color='red')
        + gg.facet_wrap(facets='Method',ncol=1,scales=scales)
        + gg.labs(
            title = 'Time Series for Residual',
            caption=caption,
        )
    )
    
    # TODO增加指示不同预测结果相差较大的区域
    
    if add_iqr_line:
        plot = (
            plot 
            + gg.geom_hline(gg.aes(linetype='"+- 1.5 * IQR"',yintercept=iqr * 1.5),color='green',size=0.5)
            + gg.geom_hline(gg.aes(linetype='"+- 1.5 * IQR"',yintercept=-iqr * 1.5),color='green',size=0.5)
            + gg.scale_linetype_manual(name=' ',values=['--','--'])
        )
    
    if time_limit:
        plot += gg.scale_x_continuous(limits=time_limit)
        #TODO 增加时间index的处理
    
    return plot