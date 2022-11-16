from numpy import genfromtxt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
print(type(stats))
print(stats.shape)
print(stats[0][1])
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
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

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(stats, overall, train_size = 0.8, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(train_input, train_target)
pred = mlr.predict(test_input)

import matplotlib.pyplot as plt
plt.scatter(test_target, pred, alpha=0.4)
plt.xlabel("Actual ")
plt.ylabel("Predicted ")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

print(mlr.coef_)
print(mlr.score(train_input, train_target))
print(mlr.score(test_input, test_target))

# from sklearn.preprocessing import PolynomialFeatures
#
# poly = PolynomialFeatures(include_bias = False)
# poly.fit(train_input)
# train_poly = poly.transform(train_input)
# print(train_poly.shape)
#
