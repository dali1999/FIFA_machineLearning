from numpy import genfromtxt
import numpy as np
import tensorflow as tf
stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
print(stats)
print(stats.shape)
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
print(overall)
print(overall.shape)

#경기출전수 appearance
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

x1_data = np.array(x1_data)
x2_data = np.array(x2_data)
x3_data = np.array(x3_data)
x4_data = np.array(x4_data)
x5_data = np.array(x5_data)
x6_data = np.array(x6_data)
x7_data = np.array(x7_data)
x8_data = np.array(x8_data)

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

hypothesis = x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8 + b



