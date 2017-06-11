import numpy as np
import scipy.optimize
import copy
import sys
import jos.sage

def estimate(ecounts, m, im = 'jos'):
    """
    estimate SAGE params for K classes
    :param ecounts: expected counts array of shape (W, K)
    :param im: the implementation to use
    """
    if len(ecounts.shape) == 1:
        ecounts = np.reshape(ecounts, (-1, 1))
    [W, K] = ecounts.shape
    eta = np.zeros((W, K))

    for k in range(K):
        if im == 'jos':
            eta[:, k] = jos.sage.estimate(ecounts[:, k], m, max_its=1)
        else:
            eta[:, k] = estimate_1class(ecounts[:, k], m)

    return eta


def estimate_1class(ecounts, m):
    """
    estimate SAGE params for one class
    """
    W = ecounts.shape[0]
    inv_tau = np.ones((W,))
    eta = np.zeros((W,))
    exp_m = np.exp(m)
    di = DeltaIterator()
    while not (di.done):
        ll = lambda x: -get_log_likelihood(x, ecounts, exp_m, inv_tau)
        grad = lambda x: -get_grad(x, ecounts, exp_m, inv_tau)
        res = scipy.optimize.minimize(
            ll, eta, method='L-BFGS-B', jac=grad, options={'disp': False})
        eta = res.x
        inv_tau = 1/(eta**2)
        di.update(eta)

    return eta


def get_log_likelihood(eta, ecounts, exp_m, inv_tau):
    #W = ecounts.shape[0]
    Ck = np.sum(ecounts, axis=0)
    #C = np.sum(Ck)
    denom = np.exp(eta) * exp_m
    l = np.sum(eta.T.dot(ecounts)) - Ck * np.log( np.sum(denom) ).T \
        - 0.5 * (eta ** 2) * inv_tau
    return l


def get_grad(eta, ecounts, exp_m, inv_tau):
    #W = ecounts.shape
    C = np.sum(ecounts, axis=0)

    denom = np.exp(eta) * exp_m
    denom_norm = (denom / denom.sum())
    beta = C * denom_norm / (C + 1e-10)
    #g = -(ecounts.sum(axis=1) - beta.dot(C) - inv_tau * eta)
    #print beta.shape, C
    #print ecounts.shape, (beta * C).shape, (inv_tau * eta).shape
    g = ecounts - beta * C - inv_tau * eta
    #print g.shape
    return g


class DeltaIterator:
    """
    Iterate until convergence (change less than thresh)
    """

    def __init__(self, max_its=20, thresh=1e-4):
        self.max_its = max_its
        self.thresh = thresh
        self.done = False
        self.count = 0
        self.prev = None
        self.eps = 1e-100
        self.done = False

    def update(self, x):
        if self.prev != None:
            change = np.linalg.norm(x - self.prev) * \
                1.0 / (self.eps + np.linalg.norm((x)))
            if change < self.thresh:
                self.done = True
        self.count += 1
        if self.count > self.max_its:
            self.done = True
        self.prev = copy.deepcopy(x)


def main():
    ecount = np.asarray([[1, 1],
                         [2, 0],
                         [4, 1],
                         [0, 2],
                         [0, 1]])

    m = np.log(np.asarray([0.2, 0.2, 0.2, 0.2, 0.2]))
    eta = estimate(ecount, m)

    print eta


if __name__ == 'main':
    main()
