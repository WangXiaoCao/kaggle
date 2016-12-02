#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


act_train_df = pd.read_csv("/home/cc/data/redhatdata/data/act_train.csv",
                           dtype={'people_id': np.str,
                                  'activity_id': np.str,
                                  'outcome': np.int8},
                           parse_dates=['date'])

act_test_df = pd.read_csv("/home/cc/data/redhatdata/data/act_test.csv",
                          dtype={'people_id': np.str,
                                'activity_id': np.str},
                          parse_dates=['date'])

people_df = pd.read_csv("/home/cc/data/redhatdata/data/people.csv",
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'char_38': np.int32},
                        parse_dates=['date'])


def intersect(l_1, l_2):
    return list(set(l_1) & set(l_2))


def get_features(train, test):
    intersecting_features = intersect(train.columns, test.columns)
    intersecting_features.remove('people_id')
    intersecting_features.remove('activity_id')
    return sorted(intersecting_features)


# # 修改日期变量
def process_date(input_df):
    df = input_df.copy()
    return (df.assign(year=lambda d: df.date.dt.year,
                      month=lambda d: df.date.dt.month,
                      day=lambda d: df.date.dt.day)
            .drop('date', axis=1))


# # 去除type
def process_activity_category(input_df):
    df = input_df.copy()
    return df.assign(activity_category=lambda df:
                     df.activity_category.str.lstrip('type ').astype(np.int32))


# # 将act中空值补全维-999
def process_activity_char(input_df, colums_range):
    df = input_df.copy()
    char_columns = ['char_' + str(i) for i in colums_range]
    return (df[char_columns].fillna('type -999')
            .apply(lambda col: col.str.lstrip('type ').astype(np.int32))
            .join(df.drop(char_columns, axis=1)))


def activities_processing(input_df):
    """
    This function combines the date, activity_category and char_*
    columns transformations.
    """
    df = input_df.copy()
    return (df.pipe(process_date)
              .pipe(process_activity_category)
              .pipe(process_activity_char, range(1, 11)))


# # 去除group
def group_process(input_df):
    df = input_df.copy()
    return df.assign(group_1=lambda df: df.group_1.str.lstrip('group ').astype(np.int32))


# # 去处people中的type（处理多分类的变量）
def process_people_cat_char(input_df, columns_range):
    df = input_df.copy()
    cat_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[cat_char_columns].apply(lambda col:
                                       col.str.lstrip('type ').astype(np.int32))
                                .join(df.drop(cat_char_columns, axis=1)))


# # 将people中布尔变量变成0/1
def process_people_bool_char(input_df, columns_range):
    df = input_df.copy()
    bool_char_columns = ['char_' + str(i) for i in columns_range]
    return (df[bool_char_columns].apply(lambda col: col.astype(np.int32))
            .join(df.drop(bool_char_columns, axis=1)))


def people_processing(input_df):
    """
    This function combines the date, group_1 and char_*
    columns (inclunding boolean and categorical ones) transformations.
    """
    df = input_df.copy()
    return (df.pipe(process_date)
              .pipe(group_process)
              .pipe(process_people_cat_char, range(1, 10))
              .pipe(process_people_bool_char, range(10, 38)))


def merge_with_people(input_df, people_df):
    """
    Merge (left) the given input dataframe with the people dataframe and
    fill the missing values with -999.
    """
    df = input_df.copy()
    return (df.merge(people_df, how='left', on='people_id',
                     left_index=True, suffixes=('_activities', '_people'))
            .fillna(-999))


# Processing pipelines

processed_people_df = people_df.pipe(people_processing)
train_df = (act_train_df.pipe(activities_processing)
                        .pipe(merge_with_people, processed_people_df))
test_df = (act_test_df.pipe(activities_processing)
                      .pipe(merge_with_people, processed_people_df))

# --------------------------------------------------------#

# Output

features_list = get_features(train_df, test_df)

print("The merged features are: ")
print('\n'.join(features_list), "\n")
print("The train dataframe head is", "\n")
print(train_df.head())
print("The test dataframe head is", "\n")
print(test_df.head())


















