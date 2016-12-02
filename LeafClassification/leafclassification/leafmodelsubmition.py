#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ------------------------1.read and pre_process data--------------------- #

# 1.1 read train.csv and test.csv
data = pd.read_csv('/home/cc/data/leafclassification/train.csv')
predict_data = pd.read_csv('/home/cc/data/leafclassification/test.csv')

# 1.2 pre_process the features
ID = data.pop('id')  # remove the column of "id" from training data
label = data.pop('species')  # remove the column of "species" from training data
predict_data.pop('id')  # remove column of "id" from test data
total_data = pd.concat([data, predict_data], ignore_index=True)  # union the train and test data
norm_x = StandardScaler()\
    .fit(data)\
    .transform(total_data)  # standardize the features
# separate the train and test data
x = norm_x[:990]
predict_x = norm_x[990:]


# 1.3 pre_process the labels
# transform the texture label to int
y = LabelEncoder().fit(label).transform(label)
# change the label to the format of one-hot
y_cat = np.zeros([990, 99], float)
for i, id in enumerate(y):
    y_cat[i, id] = 1

# 1.4 select training data and test data randomly
y = list(range(1, 990))
sample_train = random.sample(y, 700)
sample_test = set(y) - set(sample_train)

x_train = [x[i] for i in sample_train]
y_train = [y_cat[i] for i in sample_train]
x_test = [x[i] for i in sample_test]
y_test = [y_cat[i] for i in sample_test]


# ------------------------2.use Tensorflow to build neural network model--------------------- #


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0, stddev=0.1, seed=1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# create placeholder for x and y
xs = tf.placeholder(tf.float32, [None, 192])
ys = tf.placeholder(tf.float32, [None, 99])

# layer 1
w1 = weight_variable([192, 1024])
b1 = bias_variable([1024])
l1 = tf.nn.relu(tf.matmul(xs, w1) + b1)

# layer 2
w2 = weight_variable([1024, 512])
b2 = bias_variable([512])
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

# layer 2
w3 = weight_variable([512, 256])
b3 = bias_variable([256])
l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

# layer of output
w4 = weight_variable([256, 99])
b4 = bias_variable([99])
l4 = tf.matmul(l3, w4) + b4

# loss function
# cross_entropy = -tf.reduce_sum(ys * tf.log(l4))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(l4, ys))

# training by step
train1 = tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)
train2 = tf.train.AdadeltaOptimizer(0.005).minimize(cross_entropy)

# compute the accuracy
softmax_output = tf.nn.softmax(l4)
correct_prediction = tf.equal(tf.argmax(softmax_output, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# ------------------------3. create tensorflow session and train the model--------------------- #

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(20000):
    if i % 1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs: x_train, ys: y_train})
        loss = cross_entropy.eval(feed_dict={xs: x_train, ys: y_train})
        print "step %d, training accuracy %g, cross_entropy %g" % (i, train_accuracy, loss)

    train1.run(feed_dict={xs: x, ys: y_cat})

print "test accuracy %g" % accuracy.eval(feed_dict={xs: x_test, ys: y_test})

# ------------------------4.prediction--------------------- #
output = softmax_output.eval(feed_dict={xs: predict_x})

csvfile = file('/home/cc/data/leafclassification/result5.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerow(list(range(193)))
writer.writerows(output)
csvfile.close()

