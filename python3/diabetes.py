from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from numpy import genfromtxt

num_of_features = 8
train = genfromtxt('diabetes_train.csv', delimiter=",")
test = genfromtxt('diabetes_test.csv', delimiter=",")

x_train = train[:, 0:num_of_features]
y_train = train[:, [num_of_features]]

model = Sequential()
model.add(Dense(units=64, input_dim=num_of_features))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# print(model)

model.compile(loss='categorical_crossentropy',
			  optimizer='sgd',
			  metrics=['accuracy'])

# model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=5, batch_size=32)
# model.summary()
# model.get_config()