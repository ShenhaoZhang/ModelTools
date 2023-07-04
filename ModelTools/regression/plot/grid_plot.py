import pandas as pd
import matplotlib.pyplot as plt

def plot_prediction(
    data    : pd.DataFrame,
    plot_var: list,
    ci_type : str,
):
    if len(plot_var) == 1:
        nrows,ncols = 1,1
    elif len(plot_var) == 2:
        nrows,ncols = 1,1
        data[plot_var[1]] = data[plot_var[1]].astype('str')
    elif len(plot_var) == 3:
        nrows,ncols = 1,2
    elif len(plot_var) == 4:
        nrows,ncols = 2,2
    else:
        raise Exception('WRONG')
    
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols)
    
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            if len(plot_var) >= 2:
                for color in data[plot_var[1]].unique():
                    plot_data = data.loc[lambda dt:dt[plot_var[1]]==color]
                    ax.plot(plot_data[plot_var[0]],plot_data['mean'])
            else :
                ax.plot(plot_data[plot_var[0]],plot_data['mean'])


if __name__ == '__main__':
    df = pd.DataFrame({
        'x1':[1,2,3,1,2,3,1,2,3],
        'x2':[1,1,1,2,2,2,3,3,3]
    })
    df['mean'] = df.x1 * df.x2
    
    plot_prediction(
        data=df,plot_var=['x1','x2'],ci_type='mean'
    )
    plt.show()