from functools import reduce

import numpy as np
import pandas as pd

# 指定函数
# 指定向量
# 指定原数据

class DataGrid:
    def __init__(self,data:pd.DataFrame) -> None:
        self.data = data
        self.var_data = []
        self.detect_type()
    
    def detect_type(self) -> dict:
        self.var_type = self.data.dtypes.astype('str').str.replace(pat='\d+',repl='',regex=True).to_dict()
    
    def process_by_type(self,var_name) -> pd.DataFrame:
        type_method={
            'int'   : lambda x: np.mean(x).round(0),
            'float' : np.mean,
            'object': find_most_freq
        }
        var_type = self.var_type[var_name]
        var_method = type_method[var_type]
        result = pd.DataFrame({var_name:[var_method(self.data[var_name])]})
        return result
    
    def get_grid(self,**variable):
        variable = variable.copy()
        finish_var = []
        
        # 根据指定值或指定函数进行处理
        for var_name,var_method in variable.items():
            if var_name not in self.data.columns:
                continue
            else:
                finish_var.append(var_name)
            
            # 指定值
            if isinstance(var_method,(list,np.ndarray,pd.Series)):
                var_method = np.asarray(var_method)
                self.var_data.append(pd.DataFrame({var_name:var_method}))
            
            # 指定函数
            elif callable(var_method):
                var_result = var_method(self.data.loc[:,var_name])
                var_result = [var_result] if not isinstance(var_result,(list,np.ndarray)) else var_result
                self.var_data.append(pd.DataFrame({var_name:var_result}))
            
            #TODO 指定字符 函数快捷方式 std minmax quantile
            else:
                raise Exception('WRONG var_method')
            
            finish_var.append(var_name)
        
        # 将未指定的变量根据数据类型进行处理
        for var_name in self.data.columns:
            if var_name in finish_var:
                continue
            self.var_data.append(self.process_by_type(var_name))
            
        self.var_data = map(lambda df:df.assign(__key__=1),self.var_data)
        grid = reduce(lambda x,y:pd.merge(x,y,on='__key__'),self.var_data).drop('__key__',axis=1)
        
        return grid


def find_most_freq(x):
    values,counts = np.unique(x,return_counts=True)
    most_freq_value = values[counts == counts.max()]
    if isinstance(most_freq_value,np.ndarray) and len(most_freq_value)>1:
        most_freq_value = most_freq_value[0]
    return most_freq_value

if __name__ == '__main__':
    data = pd.DataFrame({
        'a':[1,2,3],
        'b':[0.3,0.2,0.1],
        'c':['x','y','z']
    })
    dg = DataGrid(data)
    print(dg.detect_type())
    print(dg.get_grid())
    
    