#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack


def reduce_dimen(dataset, column, toreplace):
    for index, i in dataset[column].duplicated(keep=False).iteritems():
        if i == False:
            dataset.set_value(index, column, toreplace)
    return dataset


# # 预处理数据
def act_data_treatment(dsname):
    dataset = dsname

    # dataset.columns是列名，这个for中的代码块目的是将所有类别型的特征都用数值表示
    for col in list(dataset.columns):
        # 如果该列是类别型的特征列
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            # 并且如果数据类型是object
            if dataset[col].dtype == 'object':
                # 那么将空值填充为type 0
                dataset[col].fillna('type 0', inplace=True)
                # 然后将type去掉，只剩下数值，并转为整型32位
                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            # 如果类型是布尔类型
            elif dataset[col].dtype == 'bool':
                # 那么将数值转换为整型8位(原来是true,false文字表示的）
                dataset[col] = dataset[col].astype(np.int8)

    # 以下是为了将时间转变成年，月，日，是否周末这四个变量，并且去掉原来的date变量
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['isweekend'] = (dataset['date'].dt.weekday >= 5).astype(int)
    dataset = dataset.drop('date', axis=1)

    # 返回的数据是修改过数据格式，并且修改了表示时间打变量的新数据
    return dataset

# # 读入数据
act_train_data = pd.read_csv("/home/cc/data/redhatdata/data/act_train.csv",
                             dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8},
                             parse_dates=['date'])
act_test_data = pd.read_csv("/home/cc/data/redhatdata/data/act_test.csv",
                            dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])
people_data = pd.read_csv("/home/cc/data/redhatdata/data/people.csv",
                          dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32},
                          parse_dates=['date'])

# # 去掉char_10这一列，axis表示对每一行做处理
act_train_data = act_train_data.drop('char_10', axis=1)
act_test_data = act_test_data.drop('char_10', axis=1)

# # 打印看看格式
# print("train data shape:" + format(act_train_data.shape))
# print("test data shape:" + format(act_test_data.shape))
# print("people data shape:" + format(people_data.shape))

# # 将三类数据都做预处理
act_train_data = act_data_treatment(act_train_data)
act_test_data = act_data_treatment(act_test_data)
people_data = act_data_treatment(people_data)

# # 根据people_id这个key，将act和peolpe两个数据left join，并依照act数据中的索引
train = act_train_data.merge(people_data, on='people_id', how='left', left_index=True)
test = act_test_data.merge(people_data, on='people_id', how='left', left_index=True)
print train[:5]

# del act_train_data
# del act_test_data
# del people_data
#
# # # 将数据根据people_id升序排列
# train = train.sort_values(['people_id'], ascending=[1])
# test = test.sort_values(['people_id'], ascending=[1])
#
# # # 把列名拿出来
# train_columns = train.columns.values
# test_columns = test.columns.values
# features = list(set(train_columns) & set(test_columns))
#
# # # 将train和test中为空的值用Na代替
# train.fillna('NA', inplace=True)
# test.fillna('NA', inplace=True)
#
# # # 把名维outcome的列拿出来赋值给y
# t = train.outcome
# # # 将原来训练集中的outcome列去掉
# train = train.drop('outcome', axis=1)
# # # 将测试与训练集union起来，并且重新建索引
# whole = pd.concat([train, test], ignore_index=True)
#
#
# # # 挑选出类别变量
# categorical = ['group_1', 'activity_category',
#                'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x',
#                'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y']
# # # 没搞清楚在做啥
# for category in categorical:
#     whole = reduce_dimen(whole, category, 9999999)
#
# # # 根据索引，再将训练与测试集区分
# X = whole[:len(train)]
# X_test = whole[len(train):]

# del train
# del test
#
# # # 将X排序，将非特征的列去除，
# X = X.sort_values(['people_id'], ascending=1)
# X = X[features].drop(['people_id', 'activity_id'], axis=1)
# X_test = X_test[features].drop(['people_id', 'activity_id'], axis=1)
#
# # # 又是取出多类别的列名，将其他列（布尔与连续）追加到not_categorical中
# categorical = ['group_1', 'activity_category',
#                'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x',
#                'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y']
# not_categorical = []
# for category in X.columns:
#     if category not in categorical:
#         not_categorical.append(category)
#
# # # 将测试与训练的特征集合并，并独热编码，扩充特征
# enc = OneHotEncoder()
# enc = enc.fit(pd.concat([X[categorical], X_test[categorical]]))
# X_cat_sparse = enc.transform(X[categorical])
# X_test_cat_sparse = enc.transform(X_test[categorical])
#
# # # 将以上特征与之前挑选出的布尔和连续变量合并成一个特征集
# X_sparse = hstack((X[not_categorical], X_cat_sparse))
# X_test_sparse = hstack((X_test[not_categorical], X_test_cat_sparse))

# print("Training data: " + format(X_sparse.shape))
# print("Test data: " + format(X_test_sparse.shape))
# print("###########")
# print("One Hot enconded Test Dataset Script")

# # 转换成矩阵
# dtrain = xgb.DMatrix(X_sparse)
# dtest = xgb.DMatrix(X_test_sparse)
#
# param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic' }
# param['nthread'] = 4
# param['eval_metric'] = 'auc'
# param['subsample'] = 0.7
# param['colsample_bytree'] = 0.7
# param['min_child_weight'] = 0
# param['booster'] = "gblinear"
#
# watchlist = [(dtrain, 'train')]
# num_round = 30
# early_stopping_rounds = 10
# bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds)
#
# ypred = bst.predict(dtest)
# print ypred[:10]
# output = pd.DataFrame({'activity_id': test['activity_id'], 'outcome': ypred})
# output.head()
# output.to_csv('/home/cc/data/redhatdata/data/without_leak.csv', index=False)








