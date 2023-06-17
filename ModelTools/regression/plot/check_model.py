import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm
from scipy.stats import gaussian_kde

plt.style.use('ggplot')
#TODO
# %config InlineBackend.figure_format = 'retina'

def plot_check_model(
    residual,
    fitted_value,
    boot_pred,
    y_name='y'
):
    y_value = fitted_value + residual
    std_residual = (residual - np.mean(residual))/np.std(residual)
    abs_std_residual = np.abs(std_residual)
    fig,ax = plt.subplots(ncols=2,nrows=2,figsize=[10,6])
    fig.tight_layout(h_pad=5,w_pad=2)
    fontdict = {'fontsize':10}
    
    smooth = lowess(residual,fitted_value)
    ax[0,0].scatter(x=fitted_value,y=residual,s=10,c='#1b6ca8')
    ax[0,0].plot(smooth[:,0],smooth[:,1],color='green')
    ax[0,0].hlines(y=0,xmin=0,xmax=1,transform=ax[0,0].get_yaxis_transform(),linestyles='--',color='black')
    ax[0,0].set_title('Linearity \nReference line should be flate and horizontal',loc='left',fontdict=fontdict)
    ax[0,0].set_xlabel('Fitted values')
    ax[0,0].set_ylabel('Residuals')

    ax[0,1].set_title('Homogeneity of Variance \nReference line should be flat and horizontal',loc='left',fontdict=fontdict)
    ax[0,1].set_xlabel('Fitted values')
    ax[0,1].set_ylabel('abs(std_residuals)')
    smooth = lowess(abs_std_residual,fitted_value)
    ax[0,1].scatter(x=fitted_value,y=abs_std_residual,s=10,c='#1b6ca8')
    ax[0,1].plot(smooth[:,0],smooth[:,1],color='green')
    
    
    ax[1,0].set_title('Normality of Residuals \nDots should fall along the line',loc='left',fontdict=fontdict)
    ax[1,0].set_xlabel('Standard Normal Distribution Quantiles')
    ax[1,0].set_ylabel('sample Quantiles')
    norm_quant,emp_quant = get_quantiles(std_residual)
    ax[1,0].plot([norm_quant.min(),norm_quant.max()],[norm_quant.min(),norm_quant.max()],c='green')
    ax[1,0].scatter(x=norm_quant,y=emp_quant,c='#1b6ca8',s=10)
    
    ax[1,1].set_title('Posterior Predictive Check \nModel-predicted lines should resemble observed data line',loc='left',fontdict=fontdict)
    ax[1,1].set_xlabel(y_name)
    ax[1,1].set_ylabel('Density')
    for pred in boot_pred:
        plot_kde(pred,ax[1,1],c='#1b6ca8',alpha=0.1)
    plot_kde(y_value,ax[1,1],c='green')
    plt.show()

def get_quantiles(x):
    x = np.sort(x)
    rank = (np.argsort(x)+1) / (len(x)+1)
    norm_quant = norm(loc=0,scale=1).ppf(rank)
    emp_quant = (x-np.mean(x))/np.std(x)
    return norm_quant,emp_quant

def plot_kde(x,ax,c,linewidth=1,alpha=1):
    x = np.sort(x)
    pdf = gaussian_kde(x).pdf(x)
    ax.plot(x,pdf,color=c,linewidth=linewidth,alpha=alpha)
    ax.ticklabel_format(style='sci',axis='y',scilimits=[-3,1],useMathText=True)