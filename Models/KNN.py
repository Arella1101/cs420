import numpy as np
import os
import sys

from data_loader import LoadDataset
from sklearn import neighbors
from sklearn.metrics import classification_report



def main():

	# data loading
	df=pd.read_csv('train.csv',header=0,sep=',') 
	df = df.sample(frac=1).reset_index(drop=True)
	x_shape, y_shape = df.shape
	train_data = df.iloc[:,1:(y_shape-1)]
	train_label = df.iloc[:,y_shape-1]

	df1=pd.read_csv('test.csv',header=0,sep=',') 
	df1 = df1.sample(frac=1).reset_index(drop=True)
	x_shape1, y_shape1 = df1.shape
	test_data = df.iloc[:,1:(y_shape1-1)]
	test_label = df.iloc[:,y_shape1-1]

    train_data = train_data.reshape(-1, 48 * 48)
    test_data = test_data.reshape(-1, 48 * 48)


    # train_model
    KNN = neighbors.KNeighborsClassifier(n_neighbors=5)
    KNN.fit(train_data, train_label)
    # get the score
    score = KNN.fit(train_data, train_label).score(test_data, test_label)


if __name__ == '__main__':
    main()