from sklearn.cluster import KMeans
import numpy as np

def k_means(X, n_clusters):
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
	return kmeans.labels_

def km_all(X_train, Y_train, X_test, Y_test):
	labels_train = k_means(X_train, 2)
	labels_test = k_means(X_test, 2)
	return (1 - np.mean(np.square(Y_train - labels_train))), (1 - np.mean(np.square(Y_test - labels_test)))