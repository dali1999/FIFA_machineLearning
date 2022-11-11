from numpy import genfromtxt
import numpy as np
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

#골수 goal
x2_data = []
for i in range(stats.shape[0]):
    x2_data.append(stats[i,1])

#어시스트수 assist
x3_data = []
for i in range(stats.shape[0]):
    x3_data.append(stats[i,2])

#spG 경기당 슛
x4_data = []
for i in range(stats.shape[0]):
    x1_data.append(stats[i,3])

#키패스 keypass
x5_data = []
for i in range(stats.shape[0]):
    x1_data.append(stats[i,4])

#드리블성공 dribble
x6_data = []
for i in range(stats.shape[0]):
    x1_data.append(stats[i,5])

#피파울 fouled
x7_data = []
for i in range(stats.shape[0]):
    x1_data.append(stats[i,0])
print(x1_data)

#명성 reputation
x1_data = []
for i in range(stats.shape[0]):
    x1_data.append(stats[i,0])
print(x1_data)