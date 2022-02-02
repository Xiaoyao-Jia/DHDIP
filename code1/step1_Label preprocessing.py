'''
-*- ecoding: utf-8 -*-
@ModuleName: step1_Label preprocessing
@Author:XYJ
@Time: 2021/7/18 12:10
'''
import numpy as np
import pandas as pd


Y_df = pd.read_csv('../data/meinian_round1_train_20180408.csv', encoding='GBK')
# Y_df=Y_df.sample(frac=0.01)
print(Y_df.info())

print('Before process, Y length: {}'.format(Y_df.shape[0]))
print(Y_df['收缩压'].unique())
Y_df = Y_df.loc[~Y_df['收缩压'].isin(['未查', '弃查', '0'])]

print(Y_df['舒张压'].unique())
Y_df = Y_df.loc[~Y_df['舒张压'].isin(['未查', '弃查', '0'])]
Y_df.loc[Y_df['舒张压'] == '100164', '舒张压'] = '164'
Y_df = Y_df.loc[~Y_df['舒张压'].isin(['974'])]

print(Y_df['血清甘油三酯'].unique())
Y_df['血清甘油三酯'] = Y_df['血清甘油三酯'].apply(lambda line: "".join(
    [i for i in line if i not in ['>', ' ', '+', '轻', '度', '乳', '糜']]))
Y_df.loc[Y_df['血清甘油三酯'] == '2.2.8', '血清甘油三酯'] = '2.28'


Y_df['血清低密度脂蛋白'] = Y_df['血清低密度脂蛋白'].astype(str).apply(lambda line: "".join(
    [i for i in line if i not in [ '-']]))

Y_df[['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']] = Y_df[
    ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']].astype(np.float32)

Y_df.to_csv('../data/cache/Y_train.csv', index=False)
print('Y_train has been saved {}'.format('../data/cache/Y_train.csv'))
print('After process, Y length: {}'.format(Y_df.shape[0]))

print()
