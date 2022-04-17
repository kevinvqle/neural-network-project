from doctest import OutputChecker
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense: #used a random.seed to produce a random values for my array 
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # calculate the output using dot product

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

X,y = spiral_data(samples=100, classes = 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])














#understand the difference between ReLU activation and linear activation 

#ReLu function example: 
#inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
#output = []
#for i in inputs: 
 #   if i > 0:
 #       output.append(i)
 #   elif i<=0:
 #       output.append(0)
#print(output)



#weights = [[0.2, 0.8, -0.5, 1],
#           [0.5, -0.91, 0.26, -0.5],
#         [-0.26, -0.27, 0.17, 0.87]]

#biases = [2,3,0.5]

#weights2 = [[0.1, -0.14, 0.5],
 #          [-0.5,0.12,-0.33],
    #      [-0.44, 0.73, -0.13]]

#biases2 = [-1,2,-0.5]



#layer1_output = np.dot(inputs, np.array(weights).T) + biases # ultilized transpose in order to mulitply by a 4x4 matricies 
#layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
#print(layer2_output)



#weights = [[0.2,0.8,-0.5,1],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]
#            ]
#bias1 = 2
#bias2 = 3
#bias3 = 0.5








#layer_outputs = [] #output of current layer
#for neuron_weights, neuron_bias in zip(weights, biases):
 #   neuron_output = 0 # output of given neuron
 #   for n_input, weight in zip(inputs, neuron_weights):
  #      neuron_output+= n_input *weight
 #   neuron_output += neuron_bias
  #  layer_outputs.append(neuron_output)

#print(layer_outputs) #returns output
