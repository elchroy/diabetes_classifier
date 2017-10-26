from numpy import array, zeros, random, genfromtxt, array, dot, exp, log
from pdb import set_trace

# instantiate the network
num_of_features = 8
# net = Network([num_of_features, 500, 250, 1])

# Prepare the training data
train = genfromtxt('diabetes_train.csv', delimiter=",")
test = genfromtxt('diabetes_test.csv', delimiter=",")

total_training_data = len(train)

x_train = train[:, 0:num_of_features]
y_train = train[:, [num_of_features]]

sizes = [num_of_features, 10, 1]
random.seed(1)
weights = [2 * random.random((a, b)) - 1 for a, b in zip(sizes[:-1], sizes[1:]) ]

learning_rate = 0.25

def sigmoid (x):
		return 1 / (1 + exp(-x))

def sigmoid_derivative (x):
	return x * (1 - x)

def cost_function (target_y, output_a):
	part_one = target_y * log(output_a)
	part_two = (1 - target_y) * log(1 - output_a)
	return -1 * (sum(part_one + part_two) / total_training_data)


def cross_entropy (target_y, output_a):
	part_one = target_y / output_a
	part_two = (target_y - 1) / (1 - output_a)
	return part_one + part_two


for iter in xrange(100):
	output = sigmoid(sigmoid(x_train.dot(weights[0])).dot(weights[1]))
	error_derivative = cross_entropy(y_train, output)

	set_trace()

