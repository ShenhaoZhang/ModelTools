from sklearn import linear_model as lm
from sklearn import preprocessing as pr
from sklearn.pipeline import Pipeline 
from sklearn.decomposition import PCA 
# from sklearn.model_selection import 

pipe_pre_01 = [
    ('std',pr.StandardScaler()),
    ('inter',pr.PolynomialFeatures(interaction_only=True,degree=2)),
    ('sp',pr.SplineTransformer())
]

pipe_pre_02 = [
    ('std',pr.StandardScaler()),
    ('poly',pr.PolynomialFeatures()),
]

mod_OLS = ('OLS',lm.LinearRegression())

{
    'model_01_a':{
        'pipe':[('OLS',lm.LinearRegression())],
        'grid':None
    },
    'model_01_b':{
        'pipe':[*pipe_pre_02,('OLS',lm.LinearRegression())],
        'grid':{'poly__degree':[2,3]}
    },
    'model_01_c':{
        'pipe':[*pipe_pre_01,('OLS',lm.LinearRegression())],
        'grid':{'sp__degree':[2,3]}
    },
    'model_02_a':{
        'pipe':[('EN',lm.ElasticNet())],
        'grid':None
    },
    'model_02_b':{
        'pipe':[*pipe_pre_01,('EN',lm.ElasticNet())],
        'grid':None
    },
    'model_03_a':{
        'pipe':[('OLS',lm.ElasticNet())],
        'grid':None
    },
    'model_03_b':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_07':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_08':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_09':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_10':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_11':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_12':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_13':{
        'pipe':[],
        'grid':{
            ...
        }
    },
    'model_14':{
        'pipe':[],
        'grid':{
            ...
        }
    },
}