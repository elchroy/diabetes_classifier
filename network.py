from numpy import reshape, array, zeros, random, exp
from time import time

class Network(object):
	
	def __init__ (self, sizes):
		random.seed(1)
		self.sizes = sizes
		self.num_of_layers = len(sizes)
		self.size = self.num_of_layers - 1
		self.no_features = self.sizes[0]
		self.weights = [2 * random.random((a, b)) - 1 for a, b in zip(sizes[:-1], sizes[1:]) ]
		self.biases = [2 * random.random((a, 1)) - 1 for a in sizes[1:]]
		self.weight_deltas = [ zeros(w.shape) for w in self.weights]
		self.bias_deltas = [ zeros(b.shape) for b in self.biases]

	def evaluate (self, test_data, accuracy=0.0, compare=False):
		total = len(test_data)
		for x_test, y_test in test_data:
			act = reshape(x_test, (self.no_features, 1))
			for l in xrange(self.size):
				act = self.sigmoid(self.weights[l].T.dot(act) + self.biases[l])
			if round(act) == y_test:
				accuracy += 1
			if compare:
				print act, y_test
		return accuracy, total
		
	def train (self, training_data, epochs=10000000, check=10, lr=0.03, test_data=None):
		total = len(training_data)
		for iter in xrange(epochs):
			start_time = time()
			for x, y in training_data:
				activation = reshape(x, (self.no_features, 1))
				activations = [activation]
				for we, bi in zip(self.weights, self.biases):
					activation = self.sigmoid(we.T.dot(activation) + bi)
					activations.append(activation)

				error_derivative = self.cost_function_derivative(activations[-1], y)
				delta = error_derivative * self.sigmoid_derivative(activations[-1])
				
				for l in xrange(self.size):
					self.bias_deltas[-l-1] = delta
					self.weight_deltas[-l-1] = activations[-l-2].dot(delta.T)
					delta = (self.weights[-l-1] * self.sigmoid_derivative(activations[-l-2])).dot(delta)

				for l in xrange(self.size):
					self.biases[l] = self.biases[l] - ((lr/total) * self.bias_deltas[l])
					self.weights[l] = self.weights[l] - ((lr/total) * self.weight_deltas[l])

			if iter % check == 0:
				accuracy = 0.0
				if test_data != None:
					accuracy, total = self.evaluate(test_data, compare=True)
				else:
					accuracy, total = self.evaluate(training_data)

				print "Accuracy: {0}/{1} => {2}%".format(accuracy, total, 100*(accuracy/total))
				print "After {0} in {1} seconds\n".format(iter, time() - start_time)

	def sigmoid (self, x):
		return 1 / (1 + exp(-x))

	def sigmoid_derivative (self, x):
		return x * (1 - x)

	def cost_function_derivative (self, output, target):
		# This is the derivative of the Mean-Square Error Cost Function
		return output - target