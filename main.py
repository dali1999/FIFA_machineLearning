from numpy import genfromtxt
stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
print(stats)
print(stats.shape)
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
print(overall)
print(overall.shape)

#경기출전수 appearance
x1_data = np.array()
for i in range(len(stats)+1):
    x1_data[i] = stats[:,0]

print(x1_data)

전성현의 작업물입니다. 다른 사람들한테는 안뜨지요