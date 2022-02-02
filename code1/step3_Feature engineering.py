'''
-*- ecoding: utf-8 -*-
@ModuleName: step3_Feature engineering
@Author:XYJ
@Time: 2021/7/20 20:31
'''
import numpy as np
from scipy.stats import mstats
import pandas as pd
from pandas.api.types import is_numeric_dtype
import re
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import catboost as cb


def filter_low_frequency(df, thresh):
    print('Removing low frequent features ...')
    print('Before: {} columns'.format(len(df.columns)))
    null_count = df.isnull().mean(axis=0)

    cols_to_drop = list(null_count[null_count > (1 - thresh)].index)
    print('After: {} columns'.format(len(df.columns) - len(cols_to_drop)))
    df_in = df[[i for i in df.columns if i not in cols_to_drop]]

    return df_in
def convert_by_field_type(df_in):
    df = df_in.astype(str)
    num_top = 20
    cols_to_convert = [i for i in df.columns if i not in ['vid']]
    cols_num, cols_text, cols_text_num, cols_cat = [], [], [], []
    cols_to_keep = []

    for i, col in enumerate(cols_to_convert):

        tmp_s = df[col]
        print('Processing {}: {}'.format(i, col))
        null_frac_before = np.mean(tmp_s.isnull())
        print('Null fraction (before): {}'.format(null_frac_before))
        tmp_counts = tmp_s.value_counts()

        tmp_counts_top = tmp_counts[:num_top]

        tmp_num_unique = np.sum(tmp_counts > 10)

        tmp_counts_num = pd.to_numeric(tmp_counts_top.index.to_series(), errors='coerce')

        tmp_counts_num_extract = tmp_counts_top.index.to_series().str.extract(
            r'([-+]?\d*\.\d+|\d+)', expand=False).astype(float).round(2)
        print('Top{} counts of field values: {}'.format(num_top, tmp_counts_top))

        if tmp_num_unique <= num_top:
            ''' Cumulative number of individual data in the column >10 count  <= 20'''
            print('It is categorical!')
            df, tmp_cols = process_cat_col(df, col, list(tmp_counts[tmp_counts > 10].index))
            cols_cat.append(col)
            cols_to_keep.extend(tmp_cols)
        elif (np.sum(tmp_counts_num.isnull()) < 0.2 * num_top or is_numeric_dtype(tmp_s)):

            '''Number of null values when the column has only numeric data < 4 Or all pure numbers'''
            print('It is numeric!')
            df[col] = process_num_col(df[col])
            cols_num.append(col)
            cols_to_keep.append(col)
        elif np.sum(tmp_counts_num_extract.notnull()) > 0.2 * num_top:
            ''' Number of non-null numeric data in the column >4'''
            print('It is numeric with text!')
            df[col + '_num'] = process_num_col(df[col])
            cols_to_keep.append(col + '_num')
            if '阴性' in list(tmp_counts_top.index):
                df[col + '_pn'] = process_positive_negative(df[col])
                cols_to_keep.append(col + '_pn')
            cols_text_num.append(col)
        else:
            print('It is text!')
            df[col + '_len'] = df[col].str.len()
            df_to_concat = process_text(df[col])
            cols_to_keep.append(col + '_len')
            if df_to_concat is not None:
                df = pd.concat((df, df_to_concat), axis=1)
                cols_to_keep.extend(df_to_concat.columns)
            cols_text.append(col)

    col_types = {'num': cols_num, 'text': cols_text, 'text_num': cols_text_num, 'cat': cols_cat}

    print({k: len(v) for k, v in col_types.items()})
    return df, col_types, cols_to_keep
def process_positive_negative(s):
    s_out = pd.Series([np.nan] * len(s))
    s_out[s.str.contains('未做|未查|弃查', na=False)] = np.nan
    s_out[s.str.contains('\+|阳性|查见|检到|检出', na=False)] = 1
    s_out[s.str.contains('\-|阴性|未查见|未检到|未检出|未见', na=False)] = 0
    s_out = s_out.astype(float)
    return s_out
def process_cat_col(df, col, values_to_onehot):
    new_cols = []
    for value in values_to_onehot:
        tmp_col = col + '_' + value+'cath'
        df[tmp_col] = (df[col] == value).astype(int)
        new_cols.append(tmp_col)
    return df, new_cols

def FullToHalf(s):
    if isinstance(s, float) or s is None:
        return s
    n = []
    for char in s:
        num = ord(char)
        if num == 12288:
            num = 32
        elif (num >= 65281 and num <= 65374):
            num -= 65248
        num = chr(num)
        n.append(num)
    return ''.join(n)

def process_num_col(s):
    s = s.apply(FullToHalf)
    s = s.str.extract('([-+]?\d*\.\d+|\d+)', expand=False).astype(float).round(2)

    s = pd.Series(mstats.winsorize(s, limits=[0.01, 0.01]))
    return s

def process_text(s):
    keywords_ch = ['糖尿病', '高血压', '血脂', '脂肪肝', '慢性胃炎', '阑尾炎', '甲肝', '肾结石',
                   '胆囊切除', '甲肝', '冠心病', '胆结石', '甲状腺', '脑梗塞', '胆囊炎', '脑溢血',
                   '早搏', '杂音', '心动过缓', '心律不齐', '心动过速']
    keywords_en = ['disease_' + str(i) for i in range(len(keywords_ch))]
    dict_out = {}
    if s.name in ['0434', '0409']:
        for i, kw in enumerate(keywords_en):
            s_out = pd.Series([np.nan] * len(s))
            s_out[s.str.contains('{}|阳性|查见|检到|检出'.format(keywords_ch[i]), na=False)] = 1
            s_out[s.str.contains('无|未查见|健康|未见', na=False)] = 0
            if np.sum(s_out) > 20:
                dict_out[s.name + '_' + kw] = s_out
        df_to_concat = pd.DataFrame(dict_out)
    elif s.name == '0113':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|清晰', na=False)] = 0
        s_out[s.str.contains('弥漫性增强|不清晰|斑点状强回声|欠清晰', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0114':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|清晰', na=False)] = 0
        s_out[s.str.contains('强回声|毛糙|增厚|伴声影', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0115':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('增强', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0116':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('结构回声|回声增强|强回声|高回声', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0117':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('无回声|强回声|高回声|回声增强', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0118':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains('无回声|强回声|高回声|回声增强', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0209':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见', na=False)] = 0
        s_out[s.str.contains('充血|鼻炎|息肉', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0215':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见', na=False)] = 0
        s_out[s.str.contains('充血|咽炎|增生', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '0912':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见|不肿大|未肿大', na=False)] = 0
        s_out[s.str.contains('结节|略大|欠光滑|甲状腺肿大', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '1001':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常', na=False)] = 0
        s_out[s.str.contains(
            '不齐|心动过缓|电轴左偏|低电压|电轴右偏|高电压|T波|心动过速|肥厚', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name == '1302':
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见异常', na=False)] = 0
        s_out[s.str.contains(
            '结膜炎|胬肉|结膜结石|素斑|裂斑|高电压|T波|心动过速|肥厚', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name in ['1330', '1316']:
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见异常', na=False)] = 0
        s_out[s.str.contains(
            '动脉硬化|动脉变细|黄斑|白内障|变性|高电压|弧形斑|视网膜病变', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name in ['1402']:
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见异常|未见狭窄|弹性良好', na=False)] = 0
        s_out[s.str.contains(
            '速度减慢|弹性降低|顺应性降低|速度增快|脑血管痉挛|略增快', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name in ['3601']:
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见异常', na=False)] = 0
        s_out[s.str.contains(
            '减少|降低|骨密度降低|疏松', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    elif s.name in ['4001']:
        s_out = pd.Series([np.nan] * len(s))
        s_out[s.str.contains('无|正常|未见异常|未见狭窄|弹性良好', na=False)] = 0
        s_out[s.str.contains(
            '轻度减弱|减弱趋势|中度减弱|重度减弱|稍硬|略增快|轻度硬化|动脉硬化|血压升高|堵塞|狭窄', na=False)] = 1
        s_out.name = s.name + '_pn'
        df_to_concat = pd.DataFrame(s_out)
    else:
        return None
    return df_to_concat

def generate_process_X(df):
    df_p = df.copy()
    df_filtered = filter_low_frequency(df_p, 0.1)

    # convert columns based on the table_field type
    df_converted, col_types, cols_to_keep = convert_by_field_type(df_filtered)
    df_out = df_converted[['vid'] + cols_to_keep]

    return df_out
def interact_feats(df):
    df_in = df.copy()
    df_in['w2h'] = df_in['2403'] / df_in['2404']
    df_in['2_top5'] = df_in['191'] * df_in['1117'] * df_in['193'] * df_in['1850'] * df_in['10004']


    return df_in
def count_location_per_vid(df):
    my_cols = list(df.columns)
    cols_six = re.findall(r'\d{6}', '{}'.format(my_cols))

    pnt_l, pnt_r = 0, 0
    col_first2_cur = cols_six[pnt_l][:2]
    tmp_list = []
    df_to_merge = pd.DataFrame()
    df_to_merge['vid'] = df['vid']
    while pnt_r != len(cols_six) - 1:
        pnt_r += 1
        if cols_six[pnt_r][:2] != col_first2_cur:
            tmp_list.append(cols_six[pnt_r - 1])
            df_to_merge['num_' + col_first2_cur] = np.sum(df[tmp_list].notnull(), axis=1)
            tmp_list = []
            pnt_l = pnt_r
            col_first2_cur = cols_six[pnt_l][:2]
        else:
            tmp_list.append(cols_six[pnt_r])
    df_to_merge['num_' + col_first2_cur] = np.sum(df[tmp_list].notnull(), axis=1)
    return df_to_merge
def create_text_feats(df):

    cols_text = ['A302', '0117', '0409', '0987', '1305', '1308', 'A201', '0201',
                 '0113', '0207', '0116', '1001', '0114', '1303', '0501', '0225',
                 '4001', '0216', '0421', '1313', '1330', '0985', '0978', '0929',
                 '0420', '0222', '0984', '0124', '0115', '1102', '0440', '0516',
                 '0426', '0119', '0912', '0954', '0983', '1316', '0423', '0123',
                 '1315', '0120', '1301', '0973', '1402', 'A202', '0209', '0975',
                 '0901', '0509', '0707', '0911', '0405', '0949', '0503', '0202',
                 '0215', '0434', '0203', '0217', '0427', '0213', '0979', '2501',
                 'A301', '3601', '0537', '0947', '0972', '0541', '0435', '0210',
                 '0118', '0539', '0715', '0546', '0208', '0974', '0101', '0436',
                 '0413', '1302', '0102', '0406', '1314', '0122', '1103', '0431', '0121']
    df_to_merge = pd.DataFrame()
    df_to_merge['vid'] = df['vid']
    # sex columns
    df_to_merge['num_woman_cols'] = np.sum(df[['0501', '0503', '0509', '0516', '0539', '0537', '0539', '0541', '0550',
                                               '0551', '0549', '0121', '0122', '0123']].notnull(), axis=1)
    df_to_merge['num_man_cols'] = np.sum(df[['0120', '0125', '0981', '0984', '0983']].notnull(), axis=1)
    keywords_ch = ['糖尿病', '高血压', '血脂', '治疗中', '肥胖', '血糖', '血压高', '血脂偏高', '血压高偏高', '冠心病',
                   '脂肪肝', '不齐', '过缓', '血管弹性', '脂'
                                              '硬化', '舒张期杂音', '收缩期杂音', '低盐', '低脂']
    keywords_en = ['disease_' + str(i) for i in range(len(keywords_ch))]
    tmp_df = df[cols_text]
    for i, col in enumerate(keywords_en):
        tmp_df_2 = tmp_df.applymap(lambda x: keywords_ch[i] in str(x))

        df_to_merge[col] = np.sum(tmp_df_2, axis=1)
    #2302
    df_to_merge['if_subhealth'] = (df['2302'] == '亚健康').astype(int)
    df_to_merge['if_ill'] = (df['2302'] == '疾病').astype(int)
    df_to_merge['if_health'] = (~(df_to_merge['if_subhealth'] | df_to_merge['if_ill'])).astype(int)


    return df_to_merge
def create_low_freq_feats(df, thresh):
    null_count = df.isnull().mean(axis=0)
    cols_to_drop = list(null_count[null_count > (1 - thresh)].index)
    df['count_low_freq'] = df[cols_to_drop].notnull().sum(axis=1)
    return df[['vid', 'count_low_freq']]

if __name__ == '__main__':
    X_raw=pd.read_csv('../data/cache/X_raw.csv')
    X_raw['vid'].astype(str)
    X_processed = generate_process_X(X_raw)
    X_processed = interact_feats(X_processed)

    X_to_merge1 = count_location_per_vid(X_raw)
    X_processed = X_processed.merge(X_to_merge1, how='left', on='vid')


    X_to_merge2 = create_text_feats(X_raw)
    X_processed = X_processed.merge(X_to_merge2, how='left', on='vid')


    X_to_merge3 = create_low_freq_feats(X_raw, 0.1)
    X_processed = X_processed.merge(X_to_merge3, how='left', on='vid')


    print(X_processed.columns)
    print('Number of columns:', len(X_processed.columns))

    Y_train = pd.read_csv('../data/cache/Y_train.csv')
    data = Y_train.merge(X_processed, how='left', on='vid')


    cat1 = cb.CatBoostRegressor()
    Y = data.iloc[:, 1:6]
    X = data.iloc[:,6:]
    selector = RFE(cat1, n_features_to_select=600, step=50).fit(X, np.log1p(np.abs(Y.iloc[:,0].values)))
    X = X[X.columns[selector.support_]]
    op_data = pd.concat([data.iloc[:,0], Y, X], axis=1)

    imputation_transformer1 = SimpleImputer(np.nan, "constant", fill_value=-2)
    data = pd.DataFrame(imputation_transformer1.fit_transform(op_data))
    data.columns = op_data.columns
    data.to_csv('../data/cache/data.csv',index=0)
    print('Done!')


print()
