import numpy as np
from numpy.linalg import solve, norm
import findMin
from scipy.optimize import approx_fprime
import utils
from neural_net import log_sum_exp

class softmaxClassifier:
    
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes
        W = np.reshape(w, (k,d))
        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1
        XW = np.dot(X, W.T)
        logK = -np.max(XW, axis=1)[:, None]
        Z = np.sum(np.exp(XW+logK), axis=1)
        # Calculate the function value
        f = - np.sum(XW[y_binary] - log_sum_exp(XW))
        # Calculate the gradient value
        g = (np.exp(XW+logK) / Z[:,None] - y_binary).T@X
        return f, g.flatten()

    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k
        self.W = np.zeros(d*k)
        self.w = self.W # because the gradient checker is implemented in a silly way
        # Initial guess
        # utils.check_gradient(self, X, y)
        (self.W, f) = findMin.findMin(self.funObj, self.W, self.maxEvals, X, y, verbose=self.verbose)
        self.W = np.reshape(self.W, (k,d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class SVM(softmaxClassifier):
    def __init__(self, maxEvals=400, verbose=1, lammy=0.01):
        self.maxEvals = maxEvals
        self.verbose = verbose
        self.lammy = lammy

    def funObj(self, w, X, y):
        n, d = X.shape
        # reshape w into kxd matrix
        w = np.reshape(w, (self.n_classes, d))

        f = 0
        g = np.zeros((self.n_classes, d))

        for i in range(n):
            for j in range(self.n_classes):
                if j == y[i]:
                    continue

                tmp = 1-np.dot(w[y[i]].T, X[i])+np.dot(w[j].T, X[i])
                if tmp > 0:
                    f += tmp
                    g[y[i]] -= X[i]
                    g[j] += X[i]

        # Regularization
        f += self.lammy/2 * np.sum(norm(w, axis=1)**2)
        g += self.lammy * w

        g = g.flatten()

        return f, g