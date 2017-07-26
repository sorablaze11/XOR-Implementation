from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential()
model.add(Dense(3, activation = 'sigmoid', input_dim = 2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'])
data = np.array(((1, 1), (1, 0), (0, 1), (0, 0)))
labels = np.array((0, 1, 1, 0))
model.fit(data, labels, epochs = 5000)
print('Enter two numbers:')
x, y = map(int, input().split())
if model.predict(np.array([[x, y]]))[0] > 0.8:
    print(1)
else:
    print(0)
