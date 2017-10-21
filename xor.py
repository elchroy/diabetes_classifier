"""
Artificial Neural Networks - Solving the X-OR problem

Accuracy: 4.0/4 => 100.0%
After 267 in 0.000866889953613 seconds
"""
from network import *

# instantiate the network
net = Network([2, 5, 1])


# Prepare the training data
x_train = array([
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1]
])
y_train = array([
	[0],
	[1],
	[1],
	[0]
])
training_data = zip(x_train, y_train)

# Train the neural network
net.train(training_data, epochs=10000000, lr=4, check=1)