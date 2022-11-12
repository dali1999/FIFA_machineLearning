from keras import models
from keras import layers
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import reuters

# (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
#      num_words=10000)
#
# print(train_labels[10])
# print(train_labels.shape)
# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
#
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#     return results
#
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# print(x_train.shape)
# print(type(x_test))
# print(x_test.shape)













stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
print(stats)
print(stats.shape)  #(688,8)
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
overall = overall - 67
print(overall)
print(overall.shape)  #(688,)

overall_encoded = np_utils.to_categorical(overall)

print(overall_encoded)


model = models.Sequential()
model.add(layers.Dense(64, input_dim= 8, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))
#
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(stats, overall_encoded, epochs=200, batch_size=1)
print("\n accuracy : %.4f" %(model.evaluate(stats, overall_encoded)[1]))



# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from keras.models import Sequential
# stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
# print(stats)
# print(stats.shape)
# overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
# print(overall)
# print(overall.shape)
#
# #overall
# y_data = []
# for i in range(overall.shape[0]):
#     y_data.append(overall[i])
#
# #경기출전수 appearance
# x1_data = []
# for i in range(stats.shape[0]):
#     x1_data.append(stats[i,0])
# print(x1_data)
#
#
# #골수 goal
# x2_data = []
# for i in range(stats.shape[0]):
#     x2_data.append(stats[i,1])
# print(x2_data)
#
# #어시스트수 assist
# x3_data = []
# for i in range(stats.shape[0]):
#     x3_data.append(stats[i,2])
# print(x3_data)
#
# #spG 경기당 슛
# x4_data = []
# for i in range(stats.shape[0]):
#     x4_data.append(stats[i,3])
# print(x4_data)
#
# #키패스 keypass
# x5_data = []
# for i in range(stats.shape[0]):
#     x5_data.append(stats[i,4])
# print(x5_data)
#
# #드리블성공 dribble
# x6_data = []
# for i in range(stats.shape[0]):
#     x6_data.append(stats[i,5])
# print(x6_data)
#
# #피파울 fouled
# x7_data = []
# for i in range(stats.shape[0]):
#     x7_data.append(stats[i,6])
# print(x7_data)
#
# #명성 reputation
# x8_data = []
# for i in range(stats.shape[0]):
#     x8_data.append(stats[i,7])
# print(x8_data)
#
# x1_data = np.array(x1_data)
# x2_data = np.array(x2_data)
# x3_data = np.array(x3_data)
# x4_data = np.array(x4_data)
# x5_data = np.array(x5_data)
# x6_data = np.array(x6_data)
# x7_data = np.array(x7_data)
# x8_data = np.array(x8_data)
#
# y_data = np.array(y_data)
#
# x1 = tf.placeholder(tf.float32)
# x2 = tf.placeholder(tf.float32)
# x3 = tf.placeholder(tf.float32)
# x4 = tf.placeholder(tf.float32)
# x5 = tf.placeholder(tf.float32)
# x6 = tf.placeholder(tf.float32)
# x7 = tf.placeholder(tf.float32)
# x8 = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
#
# w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
# w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
# w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
# w4 = tf.Variable(tf.random_normal([1]), name = 'weight4')
# w5 = tf.Variable(tf.random_normal([1]), name = 'weight5')
# w6 = tf.Variable(tf.random_normal([1]), name = 'weight6')
# w7 = tf.Variable(tf.random_normal([1]), name = 'weight7')
# w8 = tf.Variable(tf.random_normal([1]), name = 'weight8')
#
# b = tf.Variable(tf.random_normal([1]), name = 'bias')