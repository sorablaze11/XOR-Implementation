'''This is a simple implementation of XOR Neural Network using keras.
    Accuracy - 100%
    Author - sorablaze_11
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Creating the model for our Neural Network
model = Sequential()

#Adding first hidden layer with 2 input parameters, 3 neurons and setting up the sigmoid activation function.
model.add(Dense(3, activation = 'sigmoid', input_dim = 2))

#Final layer takes input and produces output
model.add(Dense(1, activation = 'sigmoid'))

# Setting up our cost function, optimizer function
model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'])

# Our datasets
data = np.array(((1, 1), (1, 0), (0, 1), (0, 0)))
labels = np.array((0, 1, 1, 0))

#Training our model
model.fit(data, labels, epochs = 4000)
print('Enter two numbers:')
x, y = map(int, input().split())
if model.predict(np.array([[x, y]]))[0] > 0.8:
    print(1)
else:
    print(0)
