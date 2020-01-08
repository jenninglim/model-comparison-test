import autograd
import autograd.numpy as np

import reltest.util as util
from reltest.estimator import Estimator
from autograd import elementwise_grad as elem_grad

import logging

class KSD_U(Estimator):
    def __init__(self, kernel):
        self.kernel = kernel

    def compute(self, log_p, ref):
        """
            Log_p: Density of the model.
            ref: Samples from the true distribution.
        """
        n = ref.shape[0]
        estimates = self.estimates(log_p, ref)
        return 1/n/(n-1)*np.sum(estimates)

    def compute_covariance(self, log_ps, ref):
        """
            log_ps: List of log density of the model.
            ref   : Samples from the true distribution.
        """
        l = len(log_ps)
        m = ref.shape[0]

        covariance = np.zeros((l, l))
        l_estimates = []
        for i in range(l):
            l_estimates = l_estimates + [self.estimates(log_ps[i], ref)]

        for i in range(l):
            for j in range(i,l):
                # Variance
                if i ==j:
                    estimates = l_estimates[i]
                    covariance[i][j] = 1/((m-1)**2)/m * np.sum(np.square(np.sum(estimates,axis=0)))
                    covariance[i][j] -= 1/(m**2)/((m-1)**2) *np.square(np.sum(estimates))#np.square(self.compute(log_ps[i],ref))
                else: # Covariance
                    estimates1 = l_estimates[i]
                    estimates2 = l_estimates[j]
                    covariance[i][j] = 1/((m-1)**2)/m * np.sum(np.sum(estimates1,axis=1)*np.sum(estimates2,axis=1))
                    covariance[i][j] -= 1/(m**2)/((m-1)**2) * np.sum(estimates1) * np.sum(estimates2)
                    covariance[j][i] = covariance[i][j]
        return 4 * (m-2)/(m-1)/m * covariance

    def n_estimates(self, n_samples):
        return n_samples

    def estimates(self, log_p, ref):
        n = ref.shape[0]
        d = ref.shape[1]

        estimates = np.zeros((n*(n-1)))
        dlog_px = log_p.grad_log(ref)
        Kxx = self.kernel.eval(ref,ref)
        ddk = self.kernel.gradXY_sum(ref,ref)

        mat2 = np.zeros((n,n))
        mat3 = np.zeros((n,n))

        mat1 = (np.matmul(dlog_px,dlog_px.T) *Kxx.T)
        ## TODO: Eigensum
        for k in range(d):
            dk_dX = self.kernel.gradX_Y(ref,ref,k)
            dk_dY = self.kernel.gradY_X(ref,ref,k)
            mat2 = mat2 + (np.repeat(dlog_px[:,k, np.newaxis],n,axis=1)*dk_dY)
            mat3 = mat3 + (np.repeat(dlog_px[:,k, np.newaxis], n,axis=1)*dk_dX.T).T
        mat4 = mat1 + mat2 + mat3 + ddk
        return mat4 - np.diag(np.diag(mat4))

class KSD_Linear(Estimator):
    """
    KSD Linear time estimator
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def compute(self, log_p, ref):
        """
            Log_p: Density of the model.
            ref: Samples from the true distribution.
        """
        n = ref.shape[0]
        estimates = self.estimates(log_p, ref)
        return 1/self.n_estimates(n)*np.sum(estimates,axis=0)

    def compute_covariance(self, log_ps, ref):
        """
            log_ps: List of log density of the model.
            ref   : Samples from the true distribution.
        """
        l = len(log_ps)
        m = ref.shape[0]
        estimatess = np.zeros((l, self.n_estimates(m)))
        for i in range(l):
            estimatess[i] = self.estimates(log_ps[i], ref)
        return np.cov(estimatess)/self.n_estimates(m)

    def n_estimates(self, n_samples):
        return int(n_samples/2)

    def estimates(self, log_p, ref):
        n = ref.shape[0]
        d = ref.shape[1]

        if n%2 == 1:
            # make it even by removing the last row
            ref = np.delete(ref, -1, axis=0)
            n = n-1
        refOdd = ref[::2,:]
        refEven = ref[1::2,:]
        estimates = np.zeros((self.n_estimates(n)))
        dlog_px = log_p.grad_log(ref)
        Kxx = self.kernel.eval(ref,ref)
        ddk = self.kernel.gradXY_sum(ref,ref)

        mat2 = np.zeros((n,n))
        mat3 = np.zeros((n,n))

        mat1 = (np.matmul(dlog_px,dlog_px.T) *Kxx.T)
        ## TODO: Eigensum
        for k in range(d):
            dk_dX = self.kernel.gradX_Y(ref,ref,k)
            dk_dY = self.kernel.gradY_X(ref,ref,k)
            mat2 = mat2 + (np.repeat(dlog_px[:,k, np.newaxis],n,axis=1)*dk_dY)
            mat3 = mat3 + (np.repeat(dlog_px[:,k, np.newaxis],n,axis=1)*dk_dX.T).T
        mat4 = mat1 + mat2 + mat3 + ddk
        e = 2*np.array(range(self.n_estimates(n)))
        o = 2*np.array(range(self.n_estimates(n)))+1
        return mat4[e,o]

def med_heuristic(ref, subsample=1000, seed=100):
    return util.meddistance(ref, subsample=subsample) **2
