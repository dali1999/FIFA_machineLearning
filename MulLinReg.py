
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig')
print(stats.shape)
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
print(overall.shape[0])

stats_train, stats_test , overall_train, overall_test = train_test_split(stats, overall, train_size = 0.8, test_size = 0.2, random_state = 42)


class MulLinearRegression:
  def __init__(self,learning_rate):
    self.w=None
    self.b=None
    self.lr=learning_rate
    self.losses=[]
    self.weight_history=[]
    self.bias_history=[]

  def forward(self,x):
    y_pred=np.sum(x*self.w)+self.b
    return y_pred

  def loss(self,x,y):
    y_pred=self.forward(x)
    return (y_pred-y)**2/overall.shape[0]
  def gradient(self,x,y):
    y_pred=self.forward(x)
    w_grad=2*x*(y_pred-y)
    b_grad=2*(y_pred-y)

    return w_grad,b_grad

  def fit(self, x_data, y_data,epochs):
    self.w = np.ones(8)  # 모델의 weight들을 전부 1로 초기화
    self.b = 0  # 모델의 bias를 0으로 초기화
    for epoch in range(epochs):
      l = 0  # 계산할 손실값
      w_grad = np.zeros(8)  # weight의 기울기를 누적할 numpy배열
      b_grad = 0  # bias의 기울기를 누적할 변수

      for x, y in zip(x_data, y_data):
        l += self.loss(x, y)
        w_i, b_i = self.gradient(x, y)

        w_grad += w_i
        b_grad += b_i

      self.w -= self.lr * (w_grad / len(y_data))
      self.b -= self.lr * (b_grad / len(y_data))

      print(
        f'epoch ({epoch + 1}) loss : {l / len(y_data):.4f} ')
      self.losses.append(l / len(y_data))
      self.weight_history.append(self.w)
      self.bias_history.append(self.b)

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

m=MulLinearRegression(learning_rate=0.0001)
m.fit(stats_train,overall_train,epochs=1000)
m.predict(stats_test,overall_test)