from altair import Undefined

def get_scales(data,x:list,y:list,type):
    x_max = data.loc[:,x].max().max()
    x_min = data.loc[:,x].min().min()
    x_lim = [x_min,x_max]
    
    y_max = data.loc[:,y].max().max()
    y_min = data.loc[:,y].min().min()
    y_lim = [y_min,y_max]
    
    if type == 'fixed':
        x_lim = x_lim
        y_lim = y_lim
    elif type == 'free':
        x_lim = Undefined
        y_lim = Undefined
    elif type == 'free_x':
        x_lim = Undefined
        y_lim = y_lim
    elif type == 'free_y':
        x_lim = x_lim
        y_lim = Undefined
    else:
        raise Exception('Wrong Type')
        
    return x_lim,y_lim