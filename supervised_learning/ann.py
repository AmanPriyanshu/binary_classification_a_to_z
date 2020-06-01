import numpy as np
import tensorflow as tf
from matplotlib import pyplot 
import time
import math

def log(n):
	#try:
	return math.log(n+1e-5)

def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic

def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

def ann_all(path, X_train, Y_train, X_test, Y_test):
	model = tf.keras.models.Sequential([tf.keras.layers.Dense(math.ceil(X_train.shape[1]/2), activation='relu'),
		tf.keras.layers.Dense(math.ceil(X_train.shape[1]/4), activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1(l=0.001))])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit(X_train, Y_train, epochs=100,verbose=0, validation_data=(X_test, Y_test))
	pyplot.plot(history.history['accuracy'])
	pyplot.plot(history.history['val_accuracy'])
	pyplot.ylim([0, 1.1])
	pyplot.ylabel('accuracy')
	pyplot.xlabel('epoch')
	pyplot.legend(['accuracy', 'val_accuracy'], loc='lower right')
	pyplot.savefig(path+'ann_acc.png')
	pyplot.clf()
	start = time.process_time()
	_, train_acc = model.evaluate(X_train, Y_train, verbose=0)
	_, test_acc = model.evaluate(X_test, Y_test, verbose=0)
	time_taken = time.process_time() - start
	predy = model.predict(X_train)
	resid = np.array([Y_train[i] - predy[i] for i in range(Y_train.shape[0])])
	ll_fit = -np.sum(np.abs(resid))
	ns_probs = [0 for _ in range(Y_train.shape[0])]
	ll_overall = -np.sum(np.array([Y_train[i] - ns_probs[i] for i in range(Y_train.shape[0])]))
	mse = np.mean(np.square(predy - Y_train))
	aic = calculate_aic(Y_train.shape[0], mse, 1)
	bic = calculate_bic(Y_train.shape[0], mse, 1)
	r2 = 1 - (ll_fit/ll_overall)

	return train_acc, test_acc, aic, bic, r2, mse, time_taken/(X_train.shape[0]+X_test.shape[0])