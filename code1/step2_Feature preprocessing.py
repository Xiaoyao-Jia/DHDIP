'''
-*- ecoding: utf-8 -*-
@ModuleName: step2_Feature preprocessing
@Author:XYJ
@Time: 2021/7/18 12:41
'''
import pandas as pd

def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df

Y_train=pd.read_csv('../data/cache/Y_train.csv',encoding='utf-8')
vid_train = list(Y_train['vid'].values)

X_1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', sep='$')
X_2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt', sep='$')
X_all = pd.concat((X_1, X_2)).reset_index(drop=True)
X_all = X_all.loc[X_all['vid'].isin(vid_train)]

table_size = X_all.groupby(['vid', 'table_id']).size().reset_index()
table_size['new_index'] = table_size['vid'] + '_' + table_size['table_id']
table_dup = table_size[table_size[0] > 1]['new_index']

X_all['new_index'] = X_all['vid'] + '_' + X_all['table_id']
dup_part = X_all[X_all['new_index'].isin(list(table_dup))]
dup_part = dup_part.sort_values(['vid', 'table_id'])
unique_part = X_all[~X_all['new_index'].isin(list(table_dup))]

X_all_dup = dup_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()
X_all_dup.rename(columns={0: 'field_results'}, inplace=True)
X_all_final = pd.concat([X_all_dup, unique_part[['vid', 'table_id', 'field_results']]])

X_p = X_all_final.pivot(index='vid', columns='table_id', values='field_results').reset_index()

# save raw_X
X_p.to_csv('../data/cache/X_raw.csv', index=False)
print('X_raw has been saved {}'.format('../data/cache/X_raw.csv'))