from network import *
import mnist_loader
from pdb import set_trace
from numpy import argmax


class HWR(Network):
	def __init__(self, sizes):
		super(HWR, self).__init__(sizes)

	def is_correct (self, act, target):
		# the index of the maximum number in the 10 dimensional output vector
		return argmax(act) == target


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = HWR([784, 30, 10])
net.train(training_data, mini_batch_size=10, test_data=test_data, check=1, lr=3.0)

# 784, 30, 10
# 
# Accuracy: 9503.0/10000 => 95.03%
# After 22 in 353.01790905 seconds
# 
# Accuracy: 9514.0/10000 => 95.14%
# After 24 in 384.076323032 second
# 
# Accuracy: 9522.0/10000 => 95.22%
# After 27 in 430.919279099 seconds
# 
# Accuracy: 9524.0/10000 => 95.24%
# After 30 in 478.280709028 seconds

# Accuracy: 9528.0/10000 => 95.28%
# After 32 in 509.073148966 seconds
# 
# Accuracy: 9536.0/10000 => 95.36%
# After 33 in 529.329262018 seconds
# 
# Accuracy: 9529.0/10000 => 95.29%
# After 34 in 547.307407141 seconds