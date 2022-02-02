'''
-*- ecoding: utf-8 -*-
@ModuleName: step4_Model establishment
@Author: XYJ
@Time: 2021/7/30 10:06
'''
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import  mean_squared_error
import xgboost
import catboost as cb
from deepforest import CascadeForestRegressor
from ngboost.ngboost import NGBoost
from sklearn.ensemble import RandomForestRegressor
#!/usr/bin/env python
# coding=utf-8

data=pd.read_csv('../data/cache/data.csv',encoding='utf-8')
# data=data.sample(frac=0.001)

Y=data.iloc[:,1:6]
X=data.iloc[:,6:]
X = sparse.csr_matrix(X.values).toarray()

mse=[]
loss =[]
for i in range(len(Y)):
    y_tr_array = np.log1p(np.abs(Y.iloc[:, i].values))
    x_train, x_test, y_train, y_test = train_test_split(X, y_tr_array, test_size=0.3,
                                                        random_state=2021)
    print('Feature'+str(i)+' CatBoost start traning：')
    cat1 = cb.CatBoostRegressor(verbose=False)
    X_train_cat = cat1.fit(x_train, y_train)
    y_test_preds_cat = cat1.predict(x_test)
    ca_test_mse_i = mean_squared_error(y_test, y_test_preds_cat)
    ca_e1_i=(((np.log(y_test_preds_cat+1)-np.log(y_test+1))**2).sum())/len(y_test)
    print(ca_e1_i)

    print('Feature'+str(i)+' LightGBM start training：')
    lgbm = LGBMRegressor()
    lgbm.fit(x_train, y_train)
    y_test_preds_lgbm = lgbm.predict(x_test)
    lg_test_mse_i = mean_squared_error(y_test, y_test_preds_lgbm)
    ca_e2_i=(((np.log(y_test_preds_lgbm+1)-np.log(y_test+1))**2).sum())/len(y_test)
    print(ca_e2_i)

    print('Featur'+str(i)+' XGBoost start training：')
    xgb = xgboost.XGBRegressor()
    xgb.fit(x_train, y_train)
    y_test_preds = xgb.predict(x_test)
    xg_test_mse_i = mean_squared_error(y_test, y_test_preds)
    ca_e3_i=(((np.log(y_test_preds+1)-np.log(y_test+1))**2).sum())/len(y_test)
    print(ca_e3_i)

    print('Feature'+str(i)+' DeepForest start training：')
    gc = CascadeForestRegressor()
    X_train_gc = gc.fit(x_train, y_train)
    y_test_preds_gc = gc.predict(x_test)
    gc_test_mse_i = mean_squared_error(y_test, y_test_preds_gc)
    ca_e4_i=(((np.log(y_test_preds_gc+1)-np.log(y_test.reshape(-1,1)+1))**2).sum())/len(y_test)
    print(ca_e4_i)

    print('Feature'+str(i)+' NGBoost start training：')
    ngb=NGBoost()
    ngb.fit(x_train, y_train)
    y_test_preds_ng = ngb.predict(x_test)
    ng_test_mse_i = mean_squared_error(y_test, y_test_preds_ng)
    ca_e5_i=(((np.log(y_test_preds_ng+1)-np.log(y_test+1))**2).sum())/len(y_test)
    print(ca_e5_i)

    print('Feature'+str(i)+' RandomForest start training：')
    rf=RandomForestRegressor()
    rf.fit(x_train, y_train)
    y_test_preds_rf = rf.predict(x_test)
    rf_test_mse_i = mean_squared_error(y_test, y_test_preds_rf)
    ca_e6_i=(((np.log(y_test_preds_rf+1)-np.log(y_test+1))**2).sum())/len(y_test)
    print(ca_e6_i)
    mse.append([round(ca_test_mse_i,4),round(lg_test_mse_i,4),round(xg_test_mse_i,4),round(gc_test_mse_i,4),round(ng_test_mse_i,4),round(rf_test_mse_i,4)])
    loss.append([round(ca_e1_i,4),round(ca_e2_i,4),round(ca_e3_i,4),round(ca_e4_i,4),round(ca_e5_i,4),round(ca_e6_i,4)])

mse = pd.DataFrame(mse)
mse.columns=['CatBoost','LightGBM','XGBoost','DeepForest','NGBoost','RandomForest']
mse.index = Y.columns
loss = pd.DataFrame(loss)
loss.columns=['CatBoost','LightGBM','XGBoost','DeepForest','NGBoost','RandomForest']
loss.index=Y.columns
print()



