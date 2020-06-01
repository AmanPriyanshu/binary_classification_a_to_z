import os
from preprocessing import preprocessing
from representation.single_feature_plot import image_plot_basic
from single_feature_analysis.logistic import logistic_sf
from supervised_learning.logistic import logistic_all
from supervised_learning.ann import ann_all
from supervised_learning.nb import nb_all
from unsupervised_learning.auto_encoders import auto_all
from unsupervised_learning.k_means import km_all
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA



def read_all_datasets(path):
	files = os.listdir(path)
	files_csv = [path+i for i in files if i.endswith('.csv')]
	return files_csv

def main_individual(path):
	files_csv = read_all_datasets(path)
	for file in files_csv:
		X, Y, X_random, Y_random, X_train, X_test, Y_train, Y_test, normal, anomalies, features = preprocessing(file)
		os.system('mkdir '+file[:-4]+'_analysis')
		ns_auc_list, lr_auc_list, train_acc_list, test_acc_list, f1_list, mse_list, aic_list, bic_list, r2_list = [], [], [], [], [], [], [], [], []
		for f_index, feature in enumerate(tqdm(features)):
			feature_value_n = normal.T[f_index]
			feature_value_a = anomalies.T[f_index]
			path_img = file[:-4]+'_analysis'
			image_plot_basic(path_img, feature_value_n, np.array([0 for i in range(feature_value_n.shape[0])]), feature_value_a, np.array([1 for i in range(feature_value_a.shape[0])]), feature)
			feature_value_train = X_train.T[f_index].reshape(-1, 1)
			feature_value_test = X_test.T[f_index].reshape(-1, 1)
			n = 1000
			ns_auc, lr_auc, lr_f1, train_acc, test_acc, mse, aic, bic, r2 = logistic_sf(feature_value_train, feature_value_test, Y_train, Y_test, path_img+'/logisitc_'+feature, n)
			ns_auc_list.append(ns_auc)
			f1_list.append(lr_f1)
			lr_auc_list.append(lr_auc)
			train_acc_list.append(train_acc)
			test_acc_list.append(test_acc)
			mse_list.append(mse)
			aic_list.append(aic)
			bic_list.append(bic)
			r2_list.append(r2)
		feature_wise = {
		"Feature Name": features,
		"Normal AUC": ns_auc_list,
		"logistic AUC": lr_auc_list,
		"R square": r2_list,
		"Train acc": train_acc_list,
		"Test acc": test_acc_list,
		"MSE": mse_list,
		"AIC": aic_list,
		"BIC": bic_list
		}
		feature_wise = pd.DataFrame(feature_wise)
		feature_wise.to_csv(file[:-4]+'_analysis/feature_wise.csv', index=False)

## FULL DATASET:
def main_full(path):
	files_csv = read_all_datasets(path)
	for file in files_csv:
		X, Y, X_random, Y_random, X_train, X_test, Y_train, Y_test, normal, anomalies, features = preprocessing(file)
		save_path = file[:-4]+'_analysis/'+'full_dataset/'
		os.system('mkdir '+save_path)
		ns_auc, lr_auc, lr_f1, train_acc, test_acc, mse, aic, bic, r2, time = logistic_all(X_train, X_test, Y_train, Y_test, save_path+'logistic', 100)
		logistic_results = {
		'ns_auc': [ns_auc],
		'lr_auc': [lr_auc],
		'lr_f1': [lr_f1],
		'train_acc': [train_acc],
		'test_acc': [test_acc],
		'mse': [mse],
		'aic': [aic],
		'bic': [bic],
		'r square': [r2],
		'Time': [time]
		}
		logistic_results = pd.DataFrame(logistic_results)
		logistic_results.to_csv(save_path+'logistic/results.csv')

		#ANN
		os.system('mkdir '+save_path+'ann/')
		train_acc, test_acc, aic, bic, r2, mse, time = ann_all(save_path+'/ann/', X_train, Y_train, X_test, Y_test)
		ann_results = {
		'train_acc': [train_acc],
		'test_acc': [test_acc],
		'AIC': [aic],
		'BIC': [bic],
		'R square': [r2],
		'MSE': [mse],
		'Time': [time]
		}
		ann_results = pd.DataFrame(ann_results)
		ann_results.to_csv(save_path+'/ann/results.csv')

		#Naive Bayes:
		os.system('mkdir '+save_path+'/naive_bayes/')
		train_acc, test_acc, aic, bic, r2, mse, time = nb_all(save_path+'/naive_bayes', X_train, Y_train, X_test, Y_test)
		nb_results = {
		'train_acc': [train_acc],
		'test_acc': [test_acc],
		'AIC': [aic],
		'BIC': [bic],
		'R square': [r2],
		'MSE': [mse],
		'Time': [time]
		}
		nb_results = pd.DataFrame(nb_results)
		nb_results.to_csv(save_path+'/naive_bayes/results.csv')

		# Auto Encoders:
		os.system('mkdir '+save_path+'/auto_encoders/')
		auto_all(save_path+'/auto_encoders', normal, anomalies)

		# K Means:
		os.system('mkdir '+save_path+'/k_means/')
		train_acc, test_acc = km_all(X_train, Y_train, X_test, Y_test)
		results = {
		'Test Acc': [test_acc],
		'Train Acc': [train_acc]
		}
		results = pd.DataFrame(results)
		results.to_csv(save_path+'/k_means/results.csv')


def feature_selection(path):
	files_csv = read_all_datasets(path)
	for file in files_csv:
		X, Y, X_random, Y_random, X_train, X_test, Y_train, Y_test, normal, anomalies, features = preprocessing(file)
		save_path = file[:-4]+'_analysis/'+'selected_features/'

		#Chi square k best:
		X = np.array([(row-np.min(row)+1e-5)/(np.max(row)-np.min(row)+1e-5) for row in X.T]).T
		X_new = SelectKBest(chi2, k=7).fit_transform(X, Y)
		k_best_features = []
		for i in X_new.T:
			for j, feature in zip(X.T, features):
				if (j[:5] == i[:5]).all():
					k_best_features.append(feature)

		model = LogisticRegression(solver='lbfgs')
		rfe = RFE(model, 5)
		fit = rfe.fit(X, Y)
		best_features = [(name, rank) for name, rank in zip(features,fit.ranking_)]

		best_feature = {
		'k_best': k_best_features,
		}
		os.system('mkdir '+save_path)
		best_feature = pd.DataFrame(best_feature)
		best_feature.to_csv(save_path+'k_best.csv')
		best_feature = {
		'rfe': best_features,
		}
		best_feature = pd.DataFrame(best_feature)
		best_feature.to_csv(save_path+'rfe.csv')


		#PCA:explained_variance_ratio_
		pca = PCA()
		pc = pca.fit_transform(X)
		featutres_pca = pca.explained_variance_ratio_
		fpca = []

		pca = {
		'PCA': featutres_pca
		}
		pca = pd.DataFrame(pca)
		pca.to_csv(save_path+'pca.csv')




os.system('clear')
path = './user_data/'
#main_individual(path)
#main_full(path)
feature_selection(path)