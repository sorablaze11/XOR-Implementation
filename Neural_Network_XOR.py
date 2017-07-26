#XOR - sorablaze_11(SUKH RAJ LIMBU)
import numpy as np

class XOR(object):
    def __init__(self, rate, train):
        #Define HyperParameters
        train = np.array(train)
        self.input_layer = 2
        self.hidden_layer = 3
        self.output_layer = 1

        #initializing acceleration rate
        self.rate = rate

        #Randomly initializing weights
        self.weights_hidden = np.random.random((self.hidden_layer, self.input_layer + 1)) # 3 x 3 Along with bias
        self.weights_output = np.random.random((1, self.hidden_layer + 1)) # 1 x 4

        #Initializating Containers for storing outputs of different layers
        self.hidden_output = np.zeros((train.shape[0], self.hidden_layer), dtype = float)
        self.activated_hidden_output = np.zeros((train.shape[0], self.hidden_layer), dtype = float)
        self.final_output = np.zeros((train.shape[0], 1), dtype = float)
        self.activated_final_output = np.zeros((train.shape[0], 1), dtype = float)
        #self.hidden_layer_delta = np.zeros(self.hidden_layer, dtype = float)

    def feedForwardFunction(self, train, answer):
        #Propagate input through NeuralNetwork
        trainingData = np.zeros((train.shape[0], self.input_layer + 1), dtype = float) # 3 x 3 input
        for i in range(train.shape[0]):
            trainingData[i][0] = 1
        trainingData[:, 1:] = train
        self.hidden_output = np.dot(trainingData, np.transpose(self.weights_hidden))
        self.activated_hidden_output = self.sigmoid(self.hidden_output)
        trainingData = np.zeros((train.shape[0], self.hidden_layer + 1), dtype = float) # 4 x 4 input
        for i in range(train.shape[0]):
            trainingData[i][0] = 1
        trainingData[:, 1:] = self.activated_hidden_output
        self.final_output = np.dot(trainingData, np.transpose(self.weights_output))
        self.activated_final_output = self.sigmoid(self.final_output)
        cost = np.sum(self.activated_final_output - answer) / train.shape[0]
        print (cost)
        return cost

    def backPropagation(self, train, answer, x):
        #BackPropagate through the network to reduce cost function
        train = np.array(train)
        answer = np.array(answer)
        y = self.feedForwardFunction(train, answer)
        if y <= 0.00001:
            print('Training is complete')
            print(self.weights_hidden)
            print(self.weights_output)
            return True
        delta_1 = 0
        delta_2 = 0
        trainingData = np.zeros((train.shape[0], self.input_layer + 1), dtype = float) # 3 x 3 input
        for i in range(train.shape[0]):
            trainingData[i][0] = 1
        trainingData[:, 1:] = train
        for i in range(train.shape[0]):
            a1 = trainingData[i].transpose()
            z2 = np.dot(self.weights_hidden, a1)
            a2 = np.zeros((z2.shape[0] + 1), dtype = float)
            a2[0] = 1
            a2[1:] = self.sigmoid(z2)
            z3 = np.dot(self.weights_output, a2)
            a3 = self.sigmoid(z3)
            error = a3 - answer[i]
            self.hidden_layer_delta = np.dot(self.weights_output.transpose()[1:], error) * self.sigmoidPrime(z2)
            delta_2 += error * a2.transpose()
            delta_1 += self.hidden_layer_delta * a1.transpose()
        weights_output_grad = delta_2 / train.shape[0]
        weights_hidden_grad = delta_1 / train.shape[0]
        self.weights_output = self.weights_output - self.rate * weights_output_grad
        self.weights_hidden = self.weights_hidden - self.rate * weights_hidden_grad
        print ('Iteration {} complete'.format(x))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))


trainingData = [[0, 0], [0 , 1], [1 , 0], [1 , 1]]
output = [[0], [1], [1], [0]]


model = XOR(.8, trainingData)
if __name__ == "__main__" :
    for i in range(10000):
        x = model.backPropagation(trainingData, output, i + 1)
        if x:
            break
