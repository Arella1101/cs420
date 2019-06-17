import numpy as np
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


# data loading
df=pd.read_csv('train.csv',header=0,sep=',') 
df = df.sample(frac=1).reset_index(drop=True)
x_shape, y_shape = df.shape
x_train = df.iloc[:,1:(y_shape-1)]
y_train = df.iloc[:,y_shape-1]

df1=pd.read_csv('test.csv',header=0,sep=',') 
df1 = df1.sample(frac=1).reset_index(drop=True)
x_shape1, y_shape1 = df1.shape
x_test = df.iloc[:,1:(y_shape1-1)]
y_test = df.iloc[:,y_shape1-1]



cRange = [2, 4,6,8]
acc_list = []
for c in cRange:

	clf = SVC(kernel='linear', C=c)
	clf.fit(x_train, y_train)
	acc = clf.score(x_test, y_test)
	print(c, acc)
	acc_list.append(acc)
	plt.figure()
	x = [cRange[i] for i in range(len(acc_list))]
	y = acc_list


kernel_list = [ 'linear', 'poly', 'rbf', 'sigmoid']
acc_list = []
for kernel in kernel_list:

	clf = SVC(kernel=kernel, C=6)
	clf.fit(x_train, y_train)

	acc = clf.score(x_test, y_test)
	print(kernel, acc)
	acc_list.append(acc)
	plt.figure()
	x = [kernel_list[i] for i in range(len(acc_list))]
	y = acc_list