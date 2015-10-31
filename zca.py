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

        zca_white = T.dot(T.dot(u, T.nlinalg.diag(1. / T.sqrt(s + eps))), u.T)
        X_zca = T.dot(X_, zca_white.T)
        self.compute_zca = theano.function([X_in, u, s, eps], X_zca)


    def fit_transform(self, X, eps):
        cov = self.sigma(X)
        u, s, _ = svd(cov)
        u = u.astype(np.float32)
        s = s.astype(np.float32)
        return self.compute_zca(X, u, s, eps)
