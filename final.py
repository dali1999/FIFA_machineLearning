from numpy import genfromtxt
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import math
tf.disable_v2_behavior()
stats = genfromtxt('hwang_stats.csv', delimiter=',', encoding='utf-8-sig')
print(type(stats))
print(stats.shape)
print(stats[0][1])
overall = genfromtxt('hwang_overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
print(overall)
print(overall.shape)




#overall
y_data = []
for i in range(overall.shape[0]):
    y_data.append(overall[i])

# #작년시즌 오버롤
x1_data = []
for i in range(stats.shape[0]):
    x1_data.append(stats[i,0])
print(x1_data)


#골수 goal
x2_data = []
for i in range(stats.shape[0]):
    x2_data.append(stats[i,1])
print(x2_data)

#어시스트수 assist
x3_data = []
for i in range(stats.shape[0]):
    x3_data.append(stats[i,2])
print(x3_data)

#spG 경기당 슛
x4_data = []
for i in range(stats.shape[0]):
    x4_data.append(stats[i,3])
print(x4_data)

#키패스 keypass
x5_data = []
for i in range(stats.shape[0]):
    x5_data.append(stats[i,4])
print(x5_data)

#드리블성공 dribble
x6_data = []
for i in range(stats.shape[0]):
    x6_data.append(stats[i,5])
print(x6_data)

#피파울 fouled
x7_data = []
for i in range(stats.shape[0]):
    x7_data.append(stats[i,6])
print(x7_data)

#명성 reputation
x8_data = []
for i in range(stats.shape[0]):
    x8_data.append(stats[i,7])
print(x8_data)




x1_train = np.array(x1_data[:400])
x2_train = np.array(x2_data[:400])
x3_train = np.array(x3_data[:400])
x4_train = np.array(x4_data[:400])
x5_train = np.array(x5_data[:400])
x6_train = np.array(x6_data[:400])
x7_train = np.array(x7_data[:400])
x8_train = np.array(x8_data[:400])

y_train = np.array(y_data[:400])

x1_test = np.array(x1_data[400:])
x2_test = np.array(x2_data[400:])
x3_test = np.array(x3_data[400:])
x4_test = np.array(x4_data[400:])
x5_test = np.array(x5_data[400:])
x6_test = np.array(x6_data[400:])
x7_test = np.array(x7_data[400:])
x8_test = np.array(x8_data[400:])

y_test = np.array(y_data[400:])

print(y_test.shape)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
x4 = tf.placeholder(tf.float32)
x5 = tf.placeholder(tf.float32)
x6 = tf.placeholder(tf.float32)
x7 = tf.placeholder(tf.float32)
x8 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
w4 = tf.Variable(tf.random_normal([1]), name = 'weight4')
w5 = tf.Variable(tf.random_normal([1]), name = 'weight5')
w6 = tf.Variable(tf.random_normal([1]), name = 'weight6')
w7 = tf.Variable(tf.random_normal([1]), name = 'weight7')
w8 = tf.Variable(tf.random_normal([1]), name = 'weight8')

b = tf.Variable(tf.random_normal([1]), name = 'bias')



hypothesis = x1_train*w1 + x2_train*w2 + x3_train*w3 + x4_train*w4 + x5_train*w5 + x6_train*w6 + x7_train*w7 + x8_train*w8 + b

cost = tf.reduce_mean(tf.square(hypothesis-y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20000):
    cost_val, hyp_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, x4:x4_data, x5:x5_data, x6:x6_data, x7:x7_data, x8:x8_data, y:y_data})
    if step % 1000 == 0:
        print(step, "cost : ", cost_val, "\nPrediction:\n", hyp_val)



result = []
count = 0

for i in range(0,176):
    hypothesis = x1_test[i] * w1 + x2_test[i] * w2 + x3_test[i] * w3 + x4_test[i] * w4 + x5_test[i] * w5 + x6_test[i] * w6 + x7_test[i] * w7 + x8_test[i] * w8 + b
    value = sess.run(hypothesis)
    if (math.floor(value) == y_test[i]):
        count += 1

print(count*100/len(y_test))
