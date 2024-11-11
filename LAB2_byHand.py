# IS LAB2
import numpy as np
import random
import matplotlib.pyplot as plt
# Run it couple times please. I have no idea why it is working the way it is. :'')
learning_rate = 0.1
max_iteration = 1000
number_of_neurons = 8
class MLP:
    def __init__(self):
        self.in_weights, self.out_weights, self.bias, self.output_bias = None, None, None, None
    
    def activation_function(self, x, function_type='sigmoid'):
        if function_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif function_type == 'tanh':
            return np.tanh(x)
    def loss_function(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)
    def backpropagation(self, inputs, targets, hidden_outputs, predicted_outputs):
        '''
        Explanations
        1- out_gradient[0] => First inputs gradient for output layer. Index determines the input
        2- hidden_gradients[0] => First index is for the input. Second index is the neuron
        3- hidden_outputs[0][0] => First index is for the input. Second index is the neuron
        '''
        '''Gradiants'''
        hidden_outputs, predicted_outputs = self.forward_pass(inputs)
        out_gradient = targets.T - predicted_outputs# Because linear
        out_gradient = out_gradient[0] # redundant array
        hidden_gradients = np.zeros((len(inputs),number_of_neurons))
        for index, x in enumerate(inputs):
            for n in range(number_of_neurons):
                hidden_gradients[index][n] = hidden_outputs[index][n] * (1 - hidden_outputs[index][n]) * self.out_weights[n] * out_gradient[n]
        hidden_gradients = np.mean(hidden_gradients,axis=1) # Since I have number_of_neurons * number_of_inputs, I am taking mean of each neurons gradient
        '''Delta weights'''
        out_weights_delta = np.zeros_like(self.out_weights)
        in_weights_delta = np.zeros_like(self.in_weights)
        for index, x in enumerate(inputs):
            for n in range(number_of_neurons):
                out_weights_delta[n] += learning_rate * out_gradient[n] * hidden_outputs[index][n]
                in_weights_delta[n] += learning_rate * hidden_gradients[index] * x[0]
        self.out_weights += out_weights_delta
        self.in_weights += in_weights_delta
    def forward_pass(self, inputs):
        hidden_outputs = np.zeros((len(inputs),number_of_neurons))
        predicted_outputs = np.zeros(len(inputs))
        for index, x in enumerate(inputs):
            for n in range(number_of_neurons):
                hidden_outputs[index][n] = self.activation_function(x[0] * self.in_weights[n] + self.bias[n])
            predicted_outputs[index] = (np.dot(hidden_outputs[index], self.out_weights) + self.output_bias)
        return hidden_outputs, predicted_outputs
    def init_weights_and_bias(self, neurons, input_size):
        in_weights = []
        out_weights = []
        bias = []
        output_bias = random.uniform(-1, 1)
        for _ in range(neurons):
            in_weights.append(random.uniform(-1, 1))
            out_weights.append(random.uniform(-1, 1))
            bias.append(random.uniform(-1, 1))
        self.in_weights, self.out_weights, self.bias, self.output_bias = in_weights, out_weights, bias, output_bias
    def train(self, inputs, targets, epochs):
        self.init_weights_and_bias(number_of_neurons, len(inputs))
        for epoch in range(epochs):
            hidden_outputs, predicted_outputs = self.forward_pass(inputs)
            self.backpropagation(inputs, targets, hidden_outputs, predicted_outputs)
            loss = self.loss_function(predicted_outputs, targets)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')
        return predicted_outputs



inputs = np.linspace(0.1, 1, 20).reshape(-1, 1)
outputs = (1 + 0.6 * np.sin(2 * np.pi * inputs / 0.7) + 0.3 * np.sin(2 * np.pi * inputs)) / 2
model = MLP()
predicted_outputs = model.train(inputs, outputs, max_iteration)

plt.scatter(inputs, outputs, color='blue', label='Actual')
plt.plot(inputs, predicted_outputs, color='red', label='Predicted')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()
#AV