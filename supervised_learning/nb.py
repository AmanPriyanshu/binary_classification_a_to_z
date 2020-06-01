import numpy as np
import tensorflow as tf
from matplotlib import pyplot 
import time
import math
from sklearn.naive_bayes import GaussianNB

def log(n):
	#try:
	return math.log(n+1e-5)

def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic

def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

def nb_all(path, X_train, Y_train, X_test, Y_test):
	
	gnb = GaussianNB()
	model = gnb.fit(X_train, Y_train)
	start = time.process_time()
	y_pred = model.predict(X_test)
	test_acc = 1-((Y_test != y_pred).sum())/Y_test.shape[0]
	y_pred = model.predict(X_train)
	train_acc = 1-((Y_train != y_pred).sum())/Y_train.shape[0]
	time_taken = time.process_time() - start
	predy = y_pred
	resid = np.array([Y_train[i] - predy[i] for i in range(Y_train.shape[0])])
	ll_fit = -np.sum(np.abs(resid))
	ns_probs = [0 for _ in range(Y_train.shape[0])]
	ll_overall = -np.sum(np.array([Y_train[i] - ns_probs[i] for i in range(Y_train.shape[0])]))
	mse = np.mean(np.square(predy - Y_train))
	aic = calculate_aic(Y_train.shape[0], mse, 1)
	bic = calculate_bic(Y_train.shape[0], mse, 1)
	r2 = 1 - (ll_fit/ll_overall)

	return train_acc, test_acc, aic, bic, r2, mse, time_taken/(X_train.shape[0]+X_test.shape[0])