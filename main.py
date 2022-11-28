import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



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

stats.shape
overall.shape

x_train = torch.FloatTensor(stats[:400])
y_train = torch.FloatTensor(overall[:400])
x_test = torch.FloatTensor(stats[400:])
y_test = torch.FloatTensor(overall[400:])

print(y_test[0])

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32, 8)
        # 마지막 출력층의 Neuron은 1개로 설정
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


model = Net(8)
# 모델을 device 에 올립니다. (cuda:0 혹은 cpu)
model.to('cpu')
model

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
num_epoch = 200

# loss 기록하기 위한 list 정의
losses = []

for epoch in range(num_epoch):
    # loss 초기화
    running_loss = 0
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        # x, y 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
        x = x.to('cpu')
        y = y.to('cpu')

        # 그라디언트 초기화 (초기화를 수행하지 않으면 계산된 그라디언트는 누적됩니다.)
        optimizer.zero_grad()

        # output 계산: model의 __call__() 함수 호출
        y_hat = model(x)

        # 손실(loss) 계산
        loss = loss_fn(y, y_hat)

        # 미분 계산
        loss.backward()

        # 경사하강법 계산 및 적용
        optimizer.step()

        # 배치별 loss 를 누적합산 합니다.
        running_loss += loss.item()

    # 누적합산된 배치별 loss값을 배치의 개수로 나누어 Epoch당 loss를 산출합니다.
    loss = running_loss / len(x_train)
    losses.append(loss)

    # 20번의 Epcoh당 출력합니다.
    if epoch % 20 == 0:
        print("{0:05d} loss = {1:.5f}".format(epoch, loss))

print("----" * 15)
print("{0:05d} loss = {1:.5f}".format(epoch, loss))

plt.figure(figsize=(14, 6))
plt.plot(losses[:100], c='darkviolet', linestyle=':')

plt.title('Losses over epoches', fontsize=15)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()


