import numpy as np
import pandas as pd
from tqdm import trange

def preprocessing(path):
	data = pd.read_csv(path)
	features = data.columns[:-1]
	data = data.values
	X = data.T[:-1]
	X = X.T
	Y = data.T[-1]

	shuffled_indexes = np.array(range(X.shape[0]))
	np.random.shuffle(shuffled_indexes)

	X_random = X[shuffled_indexes]
	Y_random = Y[shuffled_indexes]

	X_train, X_test = X_random[:int(0.8*X_random.shape[0])], X_random[int(0.8*X_random.shape[0]):]
	Y_train, Y_test = Y_random[:int(0.8*X_random.shape[0])], Y_random[int(0.8*X_random.shape[0]):]

	normal = np.array([X[index] for index in range(Y.shape[0]) if Y[index] == 0])
	anomalies = np.array([X[index] for index in range(Y.shape[0]) if Y[index] == 1])

	return X, Y, X_random, Y_random, X_train, X_test, Y_train, Y_test, normal, anomalies, features

if __name__ == '__main__':
	preprocessing('./user_data/throughput_beginning.csv')