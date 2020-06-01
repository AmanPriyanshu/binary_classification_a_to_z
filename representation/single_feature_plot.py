import matplotlib.pyplot as plt

def image_plot_basic(path_img, normal_arr, normal_ind, attack_arr, attack_ind, feature):
	plt.scatter(normal_arr, normal_ind,color = 'green', label = 'Normal', marker='*', s=100) 
	plt.scatter(attack_arr, attack_ind,color = 'red', label = 'Attack', s=100) 
	plt.ylabel('1 - Attack, 0- Normal') 
	plt.xlabel('Feature value') 
	plt.title('Attack - red, Normal - green, Feature - '+feature) 
	plt.legend(loc='lower right') 
	plt.savefig(path_img+'/'+feature+'_plot.png') 
	plt.clf()