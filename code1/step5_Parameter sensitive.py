'''
-*- ecoding: utf-8 -*-
@ModuleName: step5_Parameter sensitive
@Author: XYJ
@Time: 2021/8/2 12:01
'''
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import GridSearchCV
import catboost as cb


data=pd.read_csv('../data/cache/data.csv')
Y=data.iloc[:,1:6]
X=data.iloc[:,6:]
X = sparse.csr_matrix(X.values).toarray()
y_tr_array = np.log1p(
                       np.abs(Y.iloc[:,0].values)
                      # np.abs(Y.iloc[:,1].values)
                      # np.abs(Y.iloc[:,2].values)
                      # np.abs(Y.iloc[:,3].values)
                      # np.abs(Y.iloc[:,4].values)
                      )  # 与数据的标准化类似
x_train, x_test, y_train, y_test = train_test_split(X, y_tr_array, test_size=0.3, random_state=2021)


# '''网格搜索'''
# cb_model = cb.CatBoostRegressor(verbose=False)
# cv_params={
# 'iterations':[1500,2000,2500,3000]#2500
# , 'learning_rate':[0.01,0.02,0.03,0.04]#0.03
# ,'depth':[6,8,10,12]#8
# ,'od_wait':[70,80,90,100]#90
# ,'l2_leaf_reg':[1,2,4]#2
# ,'eval_metric':['Logloss','RMSE','MAE','CrossEntropy']#RMSE
# ,'od_type':['IncToDec','Iter']#Iter
# }
#
# cat_search = GridSearchCV(cb_model,
#                           param_grid=cv_params ,
#                           scoring='neg_mean_squared_error',
#                           # iid=False
#                           n_jobs=-1,
#                           cv=3)
#
# cat_search.fit(x_train, y_train)
#
# print(cat_search.best_params_)
# print(cat_search.best_estimator_)

'''网格搜索结果'''
other_params= {
    'iterations':2500
,'learning_rate':0.03
    ,'depth':8#8
    ,'eval_metric':'RMSE'
    ,'od_wait':90
,'l2_leaf_reg':2
    ,'od_type': 'Iter'
}
'''Default Parameters'''
cat1 = cb.CatBoostRegressor(verbose=False)
X_train_cat = cat1.fit(x_train, y_train)
y_test_preds_cat_1 = cat1.predict(x_test)
ca_test_mse_1 = mean_squared_error(y_test, y_test_preds_cat_1)
ca_e4_1=(((np.log(y_test_preds_cat_1+1)-np.log(y_test+1))**2).sum())/len(y_test)
print(ca_e4_1)

'''Optimized Parameters'''
cat2 = cb.CatBoostRegressor(**other_params,verbose=False)
X_train_cat_2 = cat2.fit(x_train, y_train)
y_test_preds_cat_2 = cat2.predict(x_test)
ca_test_mse_2 = mean_squared_error(y_test, y_test_preds_cat_2)
ca_e4_2=(((np.log(y_test_preds_cat_2+1)-np.log(y_test+1))**2).sum())/len(y_test)

print(ca_test_mse_1)
print(ca_test_mse_2)
print(ca_e4_1)
print(ca_e4_2)
print()
