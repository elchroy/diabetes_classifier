"""
Artificial Neural Networks - Classification (Appendicitis)
"""
from network import *
from numpy import genfromtxt
from sys import argv

# instantiate the network
num_of_features = 8
net = Network([num_of_features, 500, 250, 1])

# Prepare the training data
train = genfromtxt('diabetes_train.csv', delimiter=",")
test = genfromtxt('diabetes_test.csv', delimiter=",")

x_train = train[:, 0:num_of_features]
y_train = train[:, [num_of_features]]

x_test = test[:, 0:num_of_features]
y_test = test[:, [num_of_features]]

training_data = zip(x_train, y_train)
test_data = zip(x_test, y_test)

if len(argv) > 1:
	lr = float(argv[1])
else:
	lr = 0.03

# Train the neural network
net.train(training_data, mini_batch_size=17, lr=lr, check=100, test_data=test_data)

# Accuracy: 551.0/765 => 72.0261437908%
# After 640 in 973.979948044 seconds
# 
# Accuracy: 555.0/765 => 72.5490196078%
# After 780 in 1163.19382906 seconds
# 
# 0.1 -> 2000, 100, 1 worked well
# 
# 
# self.weight_deltas[-l-1] = activations[-l-2].dot(delta.T)
# (1 - ((lr * rp) / batch_total)) * 