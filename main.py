import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from neural_net import NeuralNet

from knn import KNN
from softmax import softmaxClassifier
from softmax import SVM
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelBinarizer

import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    # This is using knn
    if question == "1":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        X_valid, y_valid = valid_set
        Xtest, ytest = test_set

        allTesterrors = []

        for k in range(1, 8):
            model = KNN(k)
            model.fit(X, y)
            y_pred = model.predict(X_valid)
            allTesterrors.append(np.mean(y_pred != y_valid))

        print("Minimum validation error is ", np.min(allTesterrors))
        print("Minimum validation k is ", np.argmin(allTesterrors)+1)

    elif question == "1.1":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set
        model = KNN(4)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        print("Test error is ", np.mean(y_pred != ytest))


    elif question == "2":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        X_valid, y_valid = valid_set
        Xtest, ytest = test_set

        model = softmaxClassifier(maxEvals=400)
        model.fit(X,y)
        y_pred = model.predict(Xtest)
        print(np.mean(y_pred != ytest))
        # allTesterrors = []
        # for k in [100, 200, 300, 400, 500]:
        #     model = softmaxClassifier(maxEvals=k)
        #     model.fit(X, y)
        #     y_pred = model.predict(X_valid)
        #     allTesterrors.append(np.mean(y_pred != y_valid))

        # print("Minimum validation error is ", np.min(allTesterrors))
        # print("Minimum validation k is ", np.argmin(allTesterrors)+1)  

    elif question == "3":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        X_valid, y_valid = valid_set
        Xtest, ytest = test_set

        model = SVM(maxEvals=400)
        model.fit(X,y)
        y_pred = model.predict(Xtest)
        print("Test error =", np.mean(y_pred != ytest))

    elif question == "4":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        X_valid, y_valid = valid_set
        Xtest, ytest = test_set

        model = NeuralNet(hidden_layer_sizes=[200], lammy=0.015, max_iter=400)
        model.fit(X,y)
        y_pred = model.predict(Xtest)
        print("Test error =", np.mean(y_pred != ytest))

    else:
        print("Unknown question: %s" % question)    