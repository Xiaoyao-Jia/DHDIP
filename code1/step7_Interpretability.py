'''
-*- ecoding: utf-8 -*-
@ModuleName: step7_Interpretability
@Author: XYJ
@Time: 2021/11/7 12:01
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as cbt
import matplotlib.pyplot as plt
import shap
# !/usr/bin/env python
# coding=utf-8

data=pd.read_csv('../data/cache/data.csv')
Y=data.iloc[:,1:6]
X=data.iloc[:,6:]
# X = sparse.csr_matrix(X.values).toarray()

other_params= {
    'iterations':2500
,'learning_rate':0.03
    ,'depth':8
    ,'eval_metric':'RMSE'
    ,'od_wait':90
,'l2_leaf_reg':2
    ,'od_type': 'Iter'
}

for i in range(len(Y)):
    y_tr_array = np.log1p(np.abs(Y.iloc[:, i].values))
    x_train, x_test, y_train, y_test = train_test_split(X, y_tr_array, test_size=0.3, random_state=2021)
    cat_i = cbt.CatBoostRegressor(**other_params, verbose=0)
    cat_i.fit(x_train,y_train)
    explainer = shap.TreeExplainer(cat_i)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    plt.show()
print()