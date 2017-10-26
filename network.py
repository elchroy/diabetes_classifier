from numpy import reshape, array, zeros, random, exp, ones
from time import time
from pdb import set_trace
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
		self.bias_velocities = [ zeros(b.shape) for b in self.biases]
		self.weight_velocities = [ zeros(w.shape) for w in self.weights]

	def evaluate (self, test_data, accuracy=0.0, compare=False):
		total = len(test_data)
		# test_data = [ ( self.normalize(xt), yt) for xt, yt in test_data ]
		for x_test, y_test in test_data:
			act = reshape(x_test, (self.no_features, 1))
			for l in xrange(self.size):
				act = self.sigmoid(self.weights[l].T.dot(act) + self.biases[l])
			if self.is_correct(act, y_test):
				accuracy += 1
			if compare:
				print act, y_test
				pass
		return accuracy, total

	def is_correct (self, act, target):
		return round(act) == target

	# This is the Min-Max Normalization
	def normalize (self, xtrain):
		return (xtrain - xtrain.min()) / (xtrain.max() - xtrain.min())
		
	def standardize (self, xtrain):
		return (xtrain - xtrain.mean()) / xtrain.std()
		
	def train (self, training_data, epochs=1000, momentum_factor=0.5, rp=1.0, check=10, lr=0.03, mini_batch_size=None, test_data=None):
		total = len(training_data)
		training_data = [ ( self.normalize(xt), yt) for xt, yt in training_data ]
		if test_data != None:
			test_data = [ ( self.normalize(xt), yt) for xt, yt in test_data ]
		start_time = time()
		if mini_batch_size == None: mini_batch_size = total
		random.shuffle(training_data)
		mini_batches = [ training_data[k:k+mini_batch_size] for k in xrange(0, total, mini_batch_size)]
			
		for iter in xrange(epochs):
			for mini_batch in mini_batches:
				batch_total = len(mini_batch)
				weight_decay_factor = float(rp)/batch_total

				for x, y in mini_batch:
					# set_trace()
					activation = reshape(x, (self.no_features, 1))
					activations = [activation]
					for we, bi in zip(self.weights, self.biases):
						activation = self.sigmoid(we.T.dot(activation) + bi)
						activations.append(activation)

					error_derivative = self.cost_function_derivative(y, activations[-1])
					delta = error_derivative * self.sigmoid_derivative(activations[-1])

					for l in xrange(self.size):
						self.bias_deltas[-l-1] = delta
						self.weight_deltas[-l-1] = activations[-l-2].dot(delta.T)
						self.weight_deltas[-l-1] = self.weight_deltas[-l-1] + ((rp/total) * self.weight_deltas[-l-1])
						# self.weight_deltas[-l-1] = self.weight_deltas[-l-1] + (weight_decay_factor * self.weights[-l-1])
						delta = (self.weights[-l-1] * self.sigmoid_derivative(activations[-l-2])).dot(delta)

					for l in xrange(self.size):
						self.bias_velocities[l] = (momentum_factor * self.bias_velocities[l]) - ((lr/batch_total) * self.bias_deltas[l])
						self.weight_velocities[l] = (momentum_factor * self.weight_velocities[l]) - ((lr/batch_total) * self.weight_deltas[l])
						self.biases[l] = self.biases[l] + self.bias_velocities[l]
						self.weights[l] = self.weights[l] + self.weight_velocities[l]
						# self.weights[l] = self.weights[l] - ((lr/batch_total) * self.weight_deltas[l])
						
						# self.weight_velocities[l] = (momentum_factor * self.weight_velocities[l]) - ((lr/batch_total) * self.weight_deltas[l]) - (((lr * rp) / batch_total) * self.weight_velocities[l])
						# self.weight_velocities[l] = self.weight_velocities[l] - ((lr/batch_total) * self.weight_deltas[l])# - (((lr * rp) / batch_total) * self.weight_velocities[l])
						# self.weights[l] = self.weights[l] - ((lr/batch_total) * self.weight_deltas[l])

			if iter % check == 0:
				accuracy = 0.0
				if test_data != None:
					accuracy, test_total = self.evaluate(test_data, compare=True)
				else:
					accuracy, test_total = self.evaluate(training_data)

				print "Accuracy: {0}/{1} => {2}%".format(accuracy, test_total, 100*(accuracy/test_total))
				print "After {0} in {1} seconds\n".format(iter, time() - start_time)

		return (self.weights, self.biases)

	def sigmoid (self, x):
		return 1 / (1 + exp(-x))

	def sigmoid_derivative (self, x):
		return x * (1 - x)

	def cost_function_derivative (self, target, output):
		# This is the derivative of the Mean-Square Error Cost Function
		return output - target




	def cross_entropy (self, target_y, output_a):
		part_one = target_y / output_a
		part_two = (target_y - 1) / (1 - output_a)
		return part_one + part_two