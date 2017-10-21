"""
Artificial Neural Networks - Classification (Appendicitis)
"""
from network import *
from numpy import genfromtxt

# instantiate the network
num_of_features = 8
net = Network([num_of_features, 1000, 1])

# Prepare the training data
train = genfromtxt('diabetes_train.csv', delimiter=",")
test = genfromtxt('diabetes_test.csv', delimiter=",")

x_train = train[:, 0:num_of_features]
y_train = train[:, [num_of_features]]

training_data = zip(x_train, y_train)

# Train the neural network
net.train(training_data, mini_batch_size=17, lr=0.03, check=10)

# Accuracy: 551.0/765 => 72.0261437908%
# After 640 in 973.979948044 seconds
# 
# Accuracy: 555.0/765 => 72.5490196078%
# After 780 in 1163.19382906 seconds