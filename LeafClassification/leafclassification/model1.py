#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ------------------------1.获取训练集，并预处理特征与标签--------------------- #

# 1.1 读入数据，用pd读入的是dataframe的形式
data = pd.read_csv('/home/cc/data/leafclassification/train.csv')
predict_data = pd.read_csv('/home/cc/data/leafclassification/test.csv')

# 1.2 预处理特征数据
ID = data.pop('id')  # 在原始数据中去掉id列
label = data.pop('species')  # 在原始数据中去掉species列，剩下的就是192列特征了
predict_data.pop('id') # 在待预测数据中去除id
total_data = pd.concat([data, predict_data], ignore_index=True)  # 将原始数据与预测数据合并
norm_x = StandardScaler()\
    .fit(data)\
    .transform(total_data)  # 将所有特征正则化（均值为0），转换后dataframe形式变成了array(array())形式
# 将原始数据与待预测数据分开
x = norm_x[:990]
predict_x = norm_x[990:]

# 1.3 预处理标签数据
# 把label由文本转换成数值型的类别标签[0,1,2...98]
y = LabelEncoder().fit(label).transform(label)
# 构造损失函数，如果使用交叉熵损失，则要将label变成one-hot的形式
y_cat = np.zeros([990, 99], float)
for i, id in enumerate(y):
    y_cat[i, id] = 1

# 1.4 分割训练集与测试机
# x_train = x[300:]
# y_train = y_cat[300:]
#
# x_test = x[:300]
# y_test = y_cat[:300]

y = list(range(1, 990))
sample_train = random.sample(y, 700)
sample_test = set(y) - set(sample_train)

x_train = [x[i] for i in sample_train]
y_train = [y_cat[i] for i in sample_train]
x_test = [x[i] for i in sample_test]
y_test = [y_cat[i] for i in sample_test]


# ------------------------2.使用tensorflow搭建DNN模型--------------------- #


# 创建权重c参数的方法
def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0, stddev=0.1, seed=1)
    return tf.Variable(initial)


# 创建偏执项参数的方法
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 为输入的x,y设置占位符
xs = tf.placeholder(tf.float32, [None, 192])
ys = tf.placeholder(tf.float32, [None, 99])

# 创建第一层
w1 = weight_variable([192, 512])
b1 = bias_variable([512])
l1 = tf.nn.relu(tf.matmul(xs, w1) + b1)

# 创建第二层
w2 = weight_variable([512, 256])
b2 = bias_variable([256])
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

# 输出层
w3 = weight_variable([256, 99])
b3 = bias_variable([99])
l3 = tf.nn.softmax(tf.matmul(l2, w3) + b3)

# 损失函数
cross_entropy = -tf.reduce_sum(ys * tf.log(l3))

# 训练
train = tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)

# 计算精度（预测对的概率）
correct_prediction = tf.equal(tf.argmax(l3, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# ------------------------3.创建tensorflow会话，运行模型--------------------- #

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(1201):
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs: x_train, ys: y_train})
        loss = cross_entropy.eval(feed_dict={xs: x_train, ys: y_train})
        print "step %d, training accuracy %g, cross_entropy %g" % (i, train_accuracy, loss)
    train.run(feed_dict={xs: x, ys: y_cat})

# print "test accuracy %g" % accuracy.eval(feed_dict={xs: x_test, ys: y_test})

# ------------------------4.预测--------------------- #
output = l3.eval(feed_dict={xs: predict_x})

csvfile = file('/home/cc/data/leafclassification/result.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerow(list(range(193)))
writer.writerows(output)
csvfile.close()





