#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------- #

# 1. read original data

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


# ---------------------------------------------------------------------- #

# 2. process data

# # 补全空值并将特征转化为数值
def remedy_missing_value(inputs):
    data = inputs.copy()

    for col in list(data.columns):
        if col not in ['people_id', 'activity_id', 'date']:
            if data[col].dtype == 'object':
                data[col].fillna('type 0', inplace=True)
                data[col] = data[col].apply(lambda x: x.split(' ')[1]).astype(np.float32)
            if data[col].dtype == 'bool':
                data[col].fillna('False')
                data[col] = data[col].astype(np.float32)
    return data


# # 处理日期变量
def process_date(inputs):
    data = inputs.copy()
    data = data.assign(year=lambda d: data.date.dt.year,
                       month=lambda d: data.date.dt.month,
                       day=lambda d: data.date.dt.day,
                       isweekend=lambda d: (data.date.dt.weekday > 5).astype(float))
    data = data.drop('date', axis=1)
    return data


train = process_date(remedy_missing_value(act_train_df))
test = process_date(remedy_missing_value(act_test_df))
people = process_date(remedy_missing_value(people_df))

# # 将所有特征合并
x_data = train.merge(people, on='people_id', how='left', left_index=True)
x_test_data = test.merge(people, on='people_id', how='left', left_index=True)

# # 提取训练样本的y值与x值，去掉非特征列
y_data = act_train_df.outcome
x_data = x_data.drop('outcome', axis=1).drop('activity_id', axis=1).drop('people_id', axis=1)
x_test_data = x_test_data.drop('activity_id', axis=1).drop('people_id', axis=1)


# # 归一化
def normalize(inputs):
    data = inputs
    col_name = data.columns

    for col in list(col_name):
        max_one, min_one = data[col].max(x=0), data[col].min(x=0)
        data[col] = data[col].apply(lambda x: (x - min_one) / (max_one - min_one))
    return data


whole = pd.concat([x_data, x_test_data], ignore_index=True)
whole = normalize(whole)
x_data = whole[:len(x_data)]
x_test_data = whole[len(x_data):]

y_data = y_data[:50000]
x_data = x_data[:50000]


# print list[y_data][:5]
# print x_data[:5]

# # 转化数据格式为tensorflow能够读取的数据
x_data_arr = []
for i in x_data.values:
    x_data_arr.append(i)

x_data = np.array(x_data_arr)
print x_data.shape
# print x_data[:5]

y_data_arr = []
for i in y_data.values:
    if i == 1:
        j = [1, 0]
        y_data_arr.append(j)
    elif i == 0:
        j = [0, 1]
        y_data_arr.append(j)

y_data = np.array(y_data_arr).astype(float)  # .reshape(2197291, 1)
print y_data.shape
# print y_data[:5]

# x_data = x_data[:50000]
# y_data = y_data[:50000]
#
# x_data_pre = x_data[50000:]
# y_data_pre = y_data[50000:]
# print x_data_pre.shape()

# # ---------------------------------------------------------------------- #
#
# # # 3. 用tensorflow搭建神经网络


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0, stddev=0.1, seed=1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

xs = tf.placeholder(tf.float32, [None, 58])
ys = tf.placeholder(tf.float32, [None, 2])

w_fc1 = weight_variable([58, 200])
b_fc1 = bias_variable([200])
h_fc1 = tf.nn.relu(tf.matmul(xs, w_fc1) + b_fc1)

# w_h2 = weight_variable([200, 100])
# b_h2 = bias_variable([100])
# y_h2 = tf.nn.relu(tf.matmul(h_fc1, w_h2) + b_h2)

w_fc2 = weight_variable([200, 2])
b_fc2 = bias_variable([2])
y_fc2 = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(ys * tf.log(y_fc2))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_fc2, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(2900):

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs: x_data, ys: y_data})
        # 打印
        print "step %d, training accuracy %g" % (i, train_accuracy)
        # print sess.run(w_fc1, feed_dict={xs: x_data, ys: y_data})[:2]
        # print sess.run(h_fc11, feed_dict={xs: x_data, ys: y_data})[:2]
        # print sess.run(h_fc1, feed_dict={xs: x_data, ys: y_data})[:2]
        # print sess.run(w_fc2, feed_dict={xs: x_data, ys: y_data})[:2]
        # print sess.run(y_fc22, feed_dict={xs: x_data, ys: y_data})[:2]
        # print sess.run(y_fc2, feed_dict={xs: x_data, ys: y_data})[:2]
    train_step.run(feed_dict={xs: x_data, ys: y_data})


# print "test accuracy %g" % accuracy.eval(feed_dict={
#         xs: x_data_pre, ys: y_data_pre})



