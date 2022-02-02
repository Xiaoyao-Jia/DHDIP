'''
-*- ecoding: utf-8 -*-
@ModuleName: step6_learning curve
@Author: XYJ
@Time: 2021/11/7 0:15
'''

import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.model_selection import train_test_split

import catboost as cb

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curves(estimator1,title,  features, target, train_sizes=np.linspace(.1, 1.0, 5),ylim=None):#mse
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=10)
    if ylim is not None:
        plt.ylim(*ylim)
    train_sizes, train_scores, validation_scores = learning_curve(estimator1, features, target, train_sizes=train_sizes,
                                                                  cv=cv, scoring='neg_mean_squared_error'
    ,n_jobs=1
                                                                  )
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label='Training error')
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label='Validation error')

    plt.ylabel('error')
    plt.xlabel('Training set size')

    plt.title(title+"Learning Curve", fontsize=11)
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()
    return plt

data=pd.read_csv('../data/cache/data.csv',encoding="utf-8")
# data = data.sample(frac=0.001)
Y=data.iloc[:,1:6]
X=data.iloc[:,6:]
X = sparse.csr_matrix(X.values).toarray()
y_tr_array = np.log1p(
                       np.abs(Y.iloc[:,0].values)
#                       np.abs(Y.iloc[:,1].values)
#                       np.abs(Y.iloc[:,2].values)
                      # np.abs(Y.iloc[:,3].values)
                      # np.abs(Y.iloc[:,4].values)
                      )
x_train, x_test, y_train, y_test = train_test_split(X, y_tr_array, test_size=0.3, random_state=2021)
other_params= {
    'iterations':2500
,'learning_rate':0.03
    ,'depth':8
    ,'eval_metric':'RMSE'
    ,'od_wait':90
,'l2_leaf_reg':2
    ,'od_type': 'Iter'
}

cat2 = cb.CatBoostRegressor(**other_params,verbose=1,task_type = 'GPU')
xgb = xgboost.XGBRegressor(random_state=1)
rf=RandomForestRegressor(random_state=1)
lgbm = LGBMRegressor(random_state=1)
plot_learning_curves(cat2, 'catboost',  x_train, y_train)
plot_learning_curves(xgb, 'xgbboost',  x_train, y_train)
plot_learning_curves(lgbm, 'lightgbm',  x_train, y_train)
plot_learning_curves(rf, 'randomforest',  x_train, y_train)

print()