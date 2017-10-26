from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from numpy import genfromtxt, random, array
from pdb import set_trace

random.seed

num_of_features = 8
train = genfromtxt('diabetes_train.csv', delimiter=",")
test = genfromtxt('diabetes_test.csv', delimiter=",")

x_train = train[:, 0:num_of_features]
y_train = train[:, [num_of_features]]

x_test = array(test[:, 0:8])
y_test = array([[d] for d in test[:, 8]])


model = Sequential()
model.add(Dense(8, input_dim=num_of_features, activation='sigmoid'))
model.add(Dense(units=500, activation="sigmoid") )
model.add(Dense(units=500, activation="sigmoid") )
model.add(Dense(units=100, activation="sigmoid") )
model.add(Dense(units=1, activation='sigmoid'))

sgd = SGD(lr=0.005)

model.compile(loss='binary_crossentropy',
			  optimizer=sgd,
			  # metrics=['accuracy']
	)

# model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=2000, batch_size=17, verbose=0)
preds = model.predict_proba(x_test)
acc = 0
# print (100 * (sum([ (round(pred[0]) == out[0]) for pred, out in zip(preds, y_test) ]))/len(test))
model.summary()
model.get_config()


"""
"""