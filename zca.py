from scipy.linalg import svd
import theano
import theano.tensor as T
import numpy as np


class ZCA(object):
    def __init__(self):
        X_in = T.matrix('X_in')
        u = T.matrix('u')
        s = T.vector('s')
        eps = T.scalar('eps')

        X_ = X_in - T.mean(X_in, 0)
        sigma = T.dot(X_.T, X_) / X_.shape[0]
        self.sigma = theano.function([X_in], sigma)

        Z = T.dot(T.dot(u, T.nlinalg.diag(1. / T.sqrt(s + eps))), u.T)
        X_zca = T.dot(X_, Z.T)
        self.compute_zca = theano.function([X_in, u, s, eps], X_zca)

        self._u = None
        self._s = None


    def fit(self, X):
        cov = self.sigma(X)
        u, s, _ = svd(cov)
        self._u = u.astype(np.float32)
        self._s = s.astype(np.float32)
        del cov
        return self

    def transform(self, X, eps):
        return self.compute_zca(X, self._u, self._s, eps)


    def fit_transform(self, X, eps):
        self.fit(X)
        return self.transform(X, eps)


