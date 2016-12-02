#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# # 1. 训练数据--现在先随机制造一些数据
x_data = np.random.rand([10, 10], dtype=float)
print x_data.shape
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise
# print y_data
#
# # # 2. 定义节点
# xs = tf.placeholder(tf.float32, [None, 1])
# ys = tf.placeholder(tf.float32, [None, 1])
#
# # # 3. 定义神经层：隐藏层与预测层
# l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# predict = add_layer(l1, 10, 1, activation_function=None)
#
# # # 4. 定义损失函数
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predict),
#                                     reduction_indices=[1]))
#
# # # 5.选择优化器使损失函数最小
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# # # 6. 对所有变量进行初始化
# init = tf.initialize_all_variables()
# sess = tf.Session()
#
# # # 7. 运行
# sess.run(init)
#
# # # 8. 循环迭代
# for i in range(1000):
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 100 == 0:
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
#         print(sess.run(predict, feed_dict={xs: x_data, ys: y_data})[:10])
#
