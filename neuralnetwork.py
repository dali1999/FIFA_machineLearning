import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import genfromtxt
import matplotlib.pyplot as plt
stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)


from sklearn.model_selection import train_test_split
#train데이터, test데이터 나누기
x_train, x_test , y_train, y_test = train_test_split(stats, overall, train_size = 0.8, test_size = 0.2, random_state = 42)

#train, test데이터들을 numpy에서 tensor로 변경
x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)
y_train = torch.Tensor(y_train)


#아래와 같은 신경망을 사용해보았으나 최종으로 선택된 신경망보다
#정확도가 더 낮았다.
class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 8)

        # 마지막 출력층의 Neuron은 1개로 설정
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.output(x)
        return x



model = Net(8)
model.to('cpu')

loss_fn = nn.MSELoss() #제곱의 평균으로 로스값계산
optimizer = optim.Adam(model.parameters(), lr=0.005)    #optimizer, learning rate = 0.005
num_epoch = 400

# loss 기록하기 위한 list 정의
losses = []

for epoch in range(num_epoch):
    # loss 초기화
    running_loss = 0
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        x = x.to('cpu')
        y = y.to('cpu')
        optimizer.zero_grad()  # gradient 초기화
        predict = model(x)  # 예측오버롤값 계산
        loss = loss_fn(y, predict)  # 예측오버롤값과 실제오버롤 값의 loss 계산
        loss.backward()  # backward 진행
        optimizer.step()  # gradient descent 계산 및 적용

        running_loss += loss.item()  # loss를 합산

    loss = running_loss / len(x_train)  # loss값을 배치의 개수로 나누어서 loss를 산출
    losses.append(loss) #losses에 저장

    # 20Epoch당 출력
    if epoch % 20 == 0:
        print("{0:05d} loss = {1:.5f}".format(epoch, loss))


#epochs에 따라 변하는 loss를 그래프로 구현
print("----" * 15)
print("{0:05d} loss = {1:.5f}".format(epoch, loss))

plt.figure(figsize=(14, 6))
plt.plot(losses[:300], c='blue', linestyle='solid')

plt.title('Losses over epoches', fontsize=15)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.ylim(2,8)
plt.show()

predict = model(x_test) #테스트데이터에 대한 모델을 사용한 예측값
round_predict = torch.round(predict) #실제 오버롤은 정수이고 예측값이 소숫점자리까지 나오기때문에 반올림을 해서 자리수를 맞추는 과정
result = y_test #테스트 데이터들에 대한 실제 오버롤값

count1 = 0  #반올림한 예측값과 실제 오버롤이 정확하게 일치할 경우의 정확도
count2 = 0  #반올림 하지 않은 예측오버롤과 실제 오버롤의 차이가 1이하인 경우의 정확도
count3 = 0  #반올림 하지 않은 예측오버롤과 실제 오버롤의 차이가 2이하인 경우의 정확도
for i in range(len(x_test)):
  if(result[i] == round_predict[i]): #실제 오버롤과 반올림예측오버롤이 같으면
          count1 += 1
  if(abs(result[i] - predict[i]) <= 1): #실제 오버롤과 반올림하지 않은 예측 오버롤의 차이가 1이하이면
    count2+=1
  if(abs(result[i] - predict[i]) <= 2): #실제 오버롤과 반올림하지 않은 예측 오버롤의 차이가 2이하이면
    count3+=1

print("accuracy : " ,count1*100/len(x_test))



print("예측 오버롤과 실제 오버롤의 차이가 1이하인 경우의 accuracy : ", count2*100/len(x_test))
print("예측 오버롤과 실제 오버롤의 차이가 2이하인 경우의 accuracy : ", count3*100/len(x_test))
