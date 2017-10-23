"""
Artificial Neural Networks - Classification (Appendicitis)
"""
from network import *
from numpy import genfromtxt
from sys import argv

# instantiate the network
num_of_features = 7
net = Network([num_of_features, 3000, 30, 1])

# Prepare the training data
train = genfromtxt('appendicitis_train.csv', delimiter=",")
test = genfromtxt('appendicitis_test.csv', delimiter=",")

x_train = train[:, 0:num_of_features]
y_train = train[:, [num_of_features]]

training_data = zip(x_train, y_train)

if len(argv) > 1:
	lr = float(argv[1])
else:
	lr = 0.03

# Train the neural network
net.train(training_data, mini_batch_size=13, lr=lr, check=10)


"""
Checking after 10 epochs

> Accuracy: 93.0/104 => 89.4230769231%
> After 1220 in 33.2526700497 seconds

with 2500 sizes - 7, 2500, 1


Accuracy: 94.0/104 => 90.3846153846%
After 1710 in 55.953938961 seconds
Accuracy: 93.0/104 => 89.4230769231%
After 2170 in 72.0837550163 seconds
with 3000 hidden neurons




Accuracy: 92.0/104 => 88.4615384615%
After 1420 in 215.779101849 seconds

Accuracy: 93.0/104 => 89.4230769231%
After 2090 in 315.968469858 seconds

Accuracy: 94.0/104 => 90.3846153846%
After 3190 in 472.718219042 seconds
Shape 7, 3000, 30, 1



# With SGD
Accuracy: 95.0/104 => 91.3461538462%
After 670 in 99.6266269684 seconds
7, 3000, 30, 1, with lr=0.03 


# With some momentum, factor of 0.5
Accuracy: 96.0/104 => 92.3076923077%
After 1610 in 343.665721178 seconds

Accuracy: 97.0/104 => 93.2692307692%
After 2040 in 442.448112011 seconds

Accuracy: 98.0/104 => 94.2307692308%
After 4030 in 845.584208965 seconds
7, 3000, 30, 1


After adding Standardization and Normalization
Accuracy: 99.0/104 => 95.1923076923%
After 3360 in 725.729406118 seconds
7, 3000, 30, 1
"""