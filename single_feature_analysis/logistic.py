from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import tensorflow as tf
import os

def log(n):
	#try:
	return math.log(n+1e-5)

def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic

def calculate_bic(n, mse, num_params):
	bic = n * log(mse) + num_params * log(n)
	return bic

def logistic_sf(trainX, testX, trainy, testy, path, n):
	os.system('mkdir '+path)
	ns_probs = [0 for _ in range(testy.shape[0])]
	model = LogisticRegression(solver='lbfgs',max_iter=1e10, tol=1e-16)
	model.fit(trainX, trainy)

	# predict probabilities, tol=1
	lr_probs = model.predict_proba(testX)
	# keep probabilities for the positive outcome only
	lr_probs = lr_probs[:, 1]
	# calculate scores
	ns_auc = roc_auc_score(testy, ns_probs)
	lr_auc = roc_auc_score(testy, lr_probs)
	# calculate roc curves
	ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
	lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
	# plot the roc curve for the model
	pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
	pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
	# axis labels
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	# show the plot
	yhat = model.predict(testX)
	mse = mean_squared_error(testy, yhat)
	
	lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
	lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)

	pyplot.savefig(path+'/roc.png')
	pyplot.clf()
	sf_2(trainX, testX, trainy, testy, path)

	testX = (testX - np.min(trainX) + 1e-5)/(np.max(trainX) - np.min(trainX) + 1e-5)
	trainX = (trainX - np.min(trainX) + 1e-5)/(np.max(trainX) - np.min(trainX) + 1e-5)
	
	model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation='sigmoid')])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit(trainX, trainy, epochs=n,verbose=0, validation_data=(testX, testy))
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	predy = model.predict(trainX)
	resid = np.array([trainy[i] - predy[i] for i in range(trainy.shape[0])])
	ll_fit = -np.sum(np.abs(resid))
	ns_probs = [0 for _ in range(trainy.shape[0])]
	ll_overall = -np.sum(np.array([trainy[i] - ns_probs[i] for i in range(trainy.shape[0])]))
	mse = np.mean(np.square(predy - trainy))
	aic = calculate_aic(trainy.shape[0], mse, 1)
	bic = calculate_bic(trainy.shape[0], mse, 1)
	r2 = 1 - (ll_fit/ll_overall)

	pyplot.plot(history.history['accuracy'])
	pyplot.plot(history.history['val_accuracy'])
	pyplot.ylim([0, 1.1])
	pyplot.ylabel('accuracy')
	pyplot.xlabel('epoch')
	pyplot.legend(['accuracy', 'val_accuracy'], loc='lower right')
	pyplot.savefig(path+'/acc.png')
	pyplot.clf()

	return ns_auc, lr_auc, lr_f1, train_acc, test_acc, mse, aic, bic, r2

def sf_2(trainX, testX, trainy, testy, path):
	model = LogisticRegression(solver='lbfgs')
	model.fit(trainX, trainy)
	# predict probabilities
	lr_probs = model.predict_proba(testX)
	# keep probabilities for the positive outcome only
	lr_probs = lr_probs[:, 1]
	# predict class values
	yhat = model.predict(testX)
	lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
	lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
	# summarize scores
	# plot the precision-recall curves
	no_skill = len(testy[testy==1]) / len(testy)
	pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
	# axis labels
	pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
	# show the legend
	pyplot.legend()
	# show the plot
	pyplot.savefig(path+'/precision.png')
	pyplot.clf()