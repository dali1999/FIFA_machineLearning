from numpy import genfromtxt
stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
print(stats)
print(stats.shape)
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
print(overall)
print(overall.shape)
print('taehwan')
print('황경수')
테스트 브랜치 입니다.