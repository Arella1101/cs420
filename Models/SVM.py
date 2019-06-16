import numpy as np
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



root = './a3a.t'
path_train = root + 'train_set.npy'
trainset = np.load(path_train).astype(np.float)

path_test = root + 'test_set.npy'
testset = np.load(path_test).astype(np.float)

label_train = [trainset[i][0] for i in range(trainset.shape[0])]
data_train = [trainset[i][1:] for i in range(trainset.shape[0])]
label_train = np.array(label_train)
data_train = np.array(data_train)

label_test = [testset[i][0] for i in range(testset.shape[0])]
data_test = [testset[i][1:] for i in range(testset.shape[0])]
label_test = np.array(label_test)
data_test = np.array(data_test)

# rescale the data, use the traditional train/test split
X_train, X_test = data_train, data_test
y_train, y_test = label_train, label_test


cRange = [2, 4,6,8]
acc_list = []
for c in cRange:

	clf = SVC(kernel='linear', C=c)
	clf.fit(X_train, label_train)
	acc = clf.score(X_test, label_test)
	print(c, acc)
	acc_list.append(acc)
	plt.figure()
	x = [cRange[i] for i in range(len(acc_list))]
	y = acc_list


kernel_list = [ 'linear', 'poly', 'rbf', 'sigmoid']
acc_list = []
for kernel in kernel_list:

	clf = SVC(kernel=kernel, C=6)
	clf.fit(X_train, label_train)

	acc = clf.score(X_test, label_test)
	print(kernel, acc)
	acc_list.append(acc)
	plt.figure()
	x = [kernel_list[i] for i in range(len(acc_list))]
	y = acc_list