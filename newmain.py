
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

stats = genfromtxt('jeon_stats.csv', delimiter=',', encoding='utf-8-sig')
print(stats.shape)
overall = genfromtxt('jeon_overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
print(overall.shape[0])

stats_train, stats_test , overall_train, overall_test = train_test_split(stats, overall, train_size = 0.8, test_size = 0.2, random_state = 42)


class MultiLinear:
  def __init__(self,learning_rate=0.001):
    self.w=None #모델의 weight 벡터 self.w=(w_1,w_2)
    self.b=None #모델의 bias
    self.lr=learning_rate #모델의 학습률
    self.losses=[] #매 에포크마다 손실을 저장하기 위한 리스트
    self.weight_history=[] #매 에포크마다 계산된 weight를 저장하기 위한 리스트
    self.bias_history=[] #매 에포크마다 계산된 bias를 저장하기 위한 리스트

  def forward(self,x):
    y_pred=np.sum(x*self.w)+self.b #np.sum함수는 인자로 받은 numpy배열의 모든 원소의 합을 return합니다.
    return y_pred

  def loss(self,x,y):
    y_pred=self.forward(x)
    return (y_pred-y)**2/overall.shape[0]

  def gradient(self,x,y):
    y_pred=self.forward(x)
    w_grad=2*x*(y_pred-y)
    b_grad=2*(y_pred-y)

    return w_grad,b_grad

  def fit(self, x_data, y_data, epochs=20):
    self.w = np.ones(8)  # 모델의 weight들을 전부 1로 초기화
    self.b = 0  # 모델의 bias를 0으로 초기화
    for epoch in range(epochs):
      l = 0  # 계산할 손실값
      w_grad = np.zeros(8)  # weight의 기울기를 누적할 numpy배열
      b_grad = 0  # bias의 기울기를 누적할 변수

      for x, y in zip(x_data, y_data):
        l += self.loss(x, y)
        w_i, b_i = self.gradient(x, y)

        w_grad += w_i  # weight누적
        b_grad += b_i  # bias누적

      self.w -= self.lr * (w_grad / len(y_data))  # weight 업데이트
      self.b -= self.lr * (b_grad / len(y_data))  # bias 업데이트

      print(
        f'epoch ({epoch + 1}) loss : {l / len(y_data):.4f} | bias : {self.b:.4f}')
      self.losses.append(l / len(y_data))  # 손실값 저장
      self.weight_history.append(self.w)  # weight 배열 저장
      self.bias_history.append(self.b)  # bias값 저장

  def predict(self,x_test,y_test):
    predict=np.zeros(overall_test[0]) #예측값 저장
    for i in range(overall_test[0]):
      predict[i]=round(np.sum(self.w*x_test[i]))


    count=0
    for j in range(overall_test[0]):
      if predict[j]==y_test[j]:
        count+=1
    acc=count/overall_test[0]
    print(acc)





model=MultiLinear(learning_rate=0.0001)
model.fit(stats_train,overall_train,epochs=600)
model.predict(stats_test,overall_test)