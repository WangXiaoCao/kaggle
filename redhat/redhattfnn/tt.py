import tensorflow as tf

weights = tf.Variable(tf.random_normal([10, 5]))

a = [[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]]
b = [[1.00000000e+00, 2.58204757e-28],
     [1.00000000e+00, 0.00000000e+00],
     [0.00000000e+00, 1.00000000e+00],
     [1.00000000e+00, 0.00000000e+00],
     [1.00000000e+00, 0.00000000e+00]]
# b = [[1, 1], [1, 0], [1, 0], [0, 1]]
# b = [[0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5]]

# a = [[-15.92529297, -3.65643311],
#      [15.31280518, -3.81536865],
#      [19.31280518, -4.81536865],
#      [15.12997437, -4.71640015],
#      [-15.27072144, -3.80490112],
#      [-16.28616333, -2.74020386],
#      [2.8201046, 11.39831543],
#      [-7.8201046, 1.39831543],
#      [-3.5038147, 5.96313477],
#      [-3.60848999, -5.1784668]]

# a = tf.random_normal([5, 2], mean=0, stddev=100)


# x = tf.placeholder(tf.float32, [None, 2])
# y_fc2 = tf.nn.softmax(a)
#
# relu = tf.nn.relu(x)

y = tf.placeholder(tf.float32, [None, 2])
p = tf.placeholder(tf.float32, [None, 2])
cross_entropy = -tf.reduce_sum(y * tf.log(p))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# print sess.run(a)
# print sess.run(y_fc2)
print sess.run(cross_entropy, feed_dict={y: a, p: b})


