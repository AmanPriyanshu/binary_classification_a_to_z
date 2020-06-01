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

def auto_all(path, normal, anomaly):
	X_train = normal
	model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(int(math.ceil(X_train.shape[1]/2)), activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(int(math.ceil(X_train.shape[1]/4)), activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(int(math.ceil(X_train.shape[1]/8)), activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(int(math.ceil(X_train.shape[1]/4)), activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(int(math.ceil(X_train.shape[1])), activation='sigmoid')
	])
	model.compile(optimizer='adam', loss='mse', metrics=['mse'])
	history = model.fit(normal, normal, epochs=100,verbose=0, validation_data=(anomaly, anomaly))
	pyplot.plot(history.history['mse'])
	pyplot.plot(history.history['val_mse'])
	pyplot.ylabel('mse')
	pyplot.xlabel('epoch')
	pyplot.legend(['mse', 'val_mse'], loc='lower right')
	pyplot.savefig(path+'/loss.png')
	pyplot.clf()