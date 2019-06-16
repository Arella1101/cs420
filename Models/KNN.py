import numpy as np
import os
import sys

from data_loader import LoadDataset
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

'''
Dim: 256

Model             Val            Test
KNN(K=1)         41.26%         41.01%
KNN(K=3)         34.55%         34.61%
KNN(K=5)         34.38%         34.05%
'''

def FeatureReduction(train_data, val_data, test_data, dimension):
    pca_class = PCA(n_components=dimension)
    pca_model = pca_class.fit(train_data)
    train_data = pca_model.transform(train_data)
    val_data = pca_model.transform(val_data)
    test_data = pca_model.transform(test_data)
    return train_data, val_data, test_data

def TestModel(model, data, label, dataset_name):
    print('Testing on', dataset_name)
    pred = model.predict(data)
    print(classification_report(label, pred))
    total_num = data.shape[0]
    counter = np.zeros((total_num,), dtype=np.int32)
    counter[pred == label] = 1
    correct = np.sum(counter)
    print('Accuracy', 1.0 * correct / total_num)


def main():
    # prepare data
    print('Start loading data...')
    train_data, train_label, val_data, val_label, test_data, test_label = LoadDataset()
    #train_data = (train_data * 1.0 / 128.0) - 1
    #val_data = (val_data * 1.0 / 128.0) - 1
    #test_data = (test_data * 1.0 / 128.0) - 1
    train_data = train_data.reshape(-1, 48 * 48)
    #train_data = train_data[:18000, :]
    #train_label = train_label[:18000]
    val_data = val_data.reshape(-1, 48 * 48)
    test_data = test_data.reshape(-1, 48 * 48)
    print('Finish loading data.')
    print('train data shape', train_data.shape)
    print('val data shape', val_data.shape)
    print('test data shape', test_data.shape)

    train_data, val_data, test_data = FeatureReduction(train_data, val_data, test_data, 256)

    print('After PCA')
    print('train data shape', train_data.shape)
    print('val data shape', val_data.shape)
    print('test data shape', test_data.shape)

    # train_model
    print('Start training...')
    KNN = neighbors.KNeighborsClassifier(n_neighbors=5)
    KNN.fit(train_data, train_label)
    print('Training finished.')
    TestModel(KNN, val_data, val_label, 'val_set')
    TestModel(KNN, test_data, test_label, 'test_set')

if __name__ == '__main__':
    main()