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

x_data = train.merge(people, on='people_id', how='left', left_index=True)
x_test_data = test.merge(people, on='people_id', how='left', left_index=True)

# x_data.fillna('Na', inplace=True)

# # 提取训练样本的y值与x值
y_data = act_train_df.outcome
x_data = x_data.drop('outcome', axis=1).drop('activity_id', axis=1).drop('people_id', axis=1)

y_data = y_data[:50]
x_data = x_data[:50]

# print list[y_data][:5]
# print x_data[:5]

x_data_arr = []
for i in x_data.values:
    x_data_arr.append(i)

x_data = np.array(x_data_arr)
# print x_data.shape
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
# print y_data.shape
# print y_data[:5]


# ---------------------------------------------------------------------- #

# # 3. 用tensorflow搭建神经网络


def add_layer(inputs, in_size, out_size, activation_function=None):

    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases

    # wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    # tf.histogram_summary(layer_name + '/outputs', outputs)
    return outputs


# 定义节点
xs = tf.placeholder(tf.float32, [None, 58])
ys = tf.placeholder(tf.float32, [None, 2])
# print ys[:5]

# 定义神经层
l1 = add_layer(xs, 58, 500, activation_function=tf.nn.relu)
# l2 = add_layer(l1, 500, 100, activation_function=tf.nn.relu)
l3 = add_layer(l1, 500, 2, activation_function=tf.nn.softmax)

# 定义损失函数
# cross_entry = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(l2),
#                                            reduction_indices=[1]))
cross_entropy = -tf.reduce_sum(ys * tf.log(l3))

# 选择优化器使损失函数最小
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(l3, 1), tf.argmax(ys, 1))
# 计算正确率,用tf.cast来将true,false转换成1,0,然后计算正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 初始化变量
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# 进行迭代
for i in range(10):

    if i % 1 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs: x_data, ys: y_data})
        # 打印
        print "step %d, training accuracy %g" % (i, train_accuracy)
        print sess.run(add_layer.weights, feed_dict={xs: x_data, ys: y_data})
    train_step.run(feed_dict={xs: x_data, ys: y_data})





