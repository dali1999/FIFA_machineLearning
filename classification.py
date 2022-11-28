from keras import models
from keras import layers
from numpy import genfromtxt
import matplotlib.pyplot as plt
from keras.utils import np_utils


stats = genfromtxt('stats.csv', delimiter=',', encoding='utf-8-sig',dtype=int)
stats[:,0]-=65
print(stats)
print(stats.shape)
overall = genfromtxt('overall.csv', delimiter=',', encoding='utf-8-sig', dtype=int)
overall = overall - 65
print(overall)
print(overall.shape)

overall_encoded = np_utils.to_categorical(overall)

print(overall_encoded)

# stats_train=stats[:400]
# stats_val=stats[400:]
# overall_train=overall_encoded[:400]
# overall_val=overall_encoded[400:]

from sklearn.model_selection import train_test_split
#train데이터, test데이터 나누기
stats_train, stats_val , overall_train, overall_val = train_test_split(stats, overall_encoded, train_size = 0.8, test_size = 0.2, random_state = 42)

print(stats_train)
print(overall_train)

model = models.Sequential()
model.add(layers.Dense(64, input_dim= 8, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(30, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(stats_train, overall_train, epochs=50, batch_size=10, validation_data=(stats_val,overall_val))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\n train accuracy : %.4f" %(model.evaluate(stats_train, overall_train)[1]))
print("\n val_accuracy : %.4f" %(model.evaluate(stats_val, overall_val)[1]))

