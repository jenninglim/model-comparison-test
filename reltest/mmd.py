import numpy as np
import kgof.util as util
import freqopttest.tst as tst

import reltest.util as util
from reltest.estimator import Estimator

class MMD_Linear(Estimator):
    """
    Linear time estimator.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def estimates(self,X,Y):
        """Compute linear mmd estimator and a linear estimate of
            the uncentred 2nd moment of h(z, z'). Total cost: O(n).
            Code from https://github.com/wittawatj/interpretable-test/
        """
        kernel = self.kernel
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if n%2 == 1:
            # make it even by removing the last row
            X = np.delete(X, -1, axis=0)
            Y = np.delete(Y, -1, axis=0)

        Xodd = X[::2, :]
        Xeven = X[1::2, :]
        assert Xodd.shape[0] == Xeven.shape[0]
        Yodd = Y[::2, :]
        Yeven = Y[1::2, :]
        assert Yodd.shape[0] == Yeven.shape[0]
        # linear mmd. O(n)
        xx = kernel.pair_eval(Xodd, Xeven)
        yy = kernel.pair_eval(Yodd, Yeven)
        xo_ye = kernel.pair_eval(Xodd, Yeven)
        xe_yo = kernel.pair_eval(Xeven, Yodd)
        h = xx + yy - xo_ye - xe_yo
        return h

    def compute(self, X,Y):
        h = self.estimates(X,Y)
        return np.mean(h)

    def compute_covariance(self, models, ref):
        l = len(models)
        n = np.shape(models[0])[0]
        H_b = np.zeros((l,self.n_estimates(n)))
        for i in range(l):
            H_b[i,:] = self.estimates(models[i], ref).flatten()
        sigma = np.cov(H_b)
        return sigma/ self.n_estimates(n)

    def n_estimates(self, n):
        return int(n/2)

class MMD_U(Estimator):
    def __init__(self, kernel):
        self.kernel = kernel

    def compute(self,X,Y):
        k = self.kernel
        Kx = k.eval(X, X)
        Ky = k.eval(Y, Y)
        Kxy = k.eval(X, Y)

        nx = Kx.shape[0]
        ny = Ky.shape[0]

        xx = (np.sum(Kx) - np.sum(np.diag(Kx)))/(nx*(nx-1))
        yy = (np.sum(Ky) - np.sum(np.diag(Ky)))/(ny*(ny-1))
        xy = (np.sum(Kxy) - np.sum(np.diag(Kxy)))/(nx*(ny-1))
        mmd = xx - 2*xy + yy
        #np.testing.assert_almost_equal(mmd, mmd2)
        return mmd #xx - 2*xy + yy

    def n_estimates(self, n):
        return n

    def estimates(self, X,Y):
        k = self.kernel
        Kx = k.eval(X, X)
        Ky = k.eval(Y, Y)
        Kxy = k.eval(X, Y)

        Kx = Kx - np.diag(np.diag(Kx))
        Ky = Ky - np.diag(np.diag(Ky))
        Kxy = Kxy - np.diag(np.diag(Kxy))
        return Kx -2.*Kxy + Ky

    def compute_covariance(self, models, ref):
        """
            Computes the covariance of the MMD estimates for X,Y
            X,Y: Samples from a distribution. With shape dim x n_samples.

            models: A list of samples of length N.
                    Each element has an array of size
                    d*m. m samples from N candidate models.
            ref:    The reference distribution of size d*m.
                    m samples from the true distribution.
        """
        n = len(models)
        m = np.shape(models[0])[0]
        cov = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i !=j :
                    cov[i][j] = get_cross_covariance(models[i],models[j],ref, self.kernel)
                else:
                    _, cov[i][j] = tst.QuadMMDTest.h1_mean_var(models[i], ref, self.kernel,
                is_var_computed=True)
        # Carries a factor of 1/n?
        return cov

def get_cross_covariance(X, Y, Z, k):
    """
    Code from: https://github.com/wittawatj/kernel-mod
    Compute the covariance of the U-statistics for two MMDs
    (Bounliphone, et al. 2016, ICLR)
    Args:
        X: numpy array of shape (nx, d), sample from the model 1
        Y: numpy array of shape (ny, d), sample from the model 2
        Z: numpy array of shape (nz, d), sample from the reference
        k: a kernel object
    Returns:
        cov: covariance of two U stats
    """
    Kzz = k.eval(Z, Z)
    # Kxx
    Kzx = k.eval(Z, X)
    # Kxy
    Kzy = k.eval(Z, Y)
    # Kxz
    Kzznd = Kzz - np.diag(np.diag(Kzz))
    # Kxxnd = Kxx-diag(diag(Kxx));

    nz = Kzz.shape[0]
    nx = Kzx.shape[1]
    ny = Kzy.shape[1]
    # m = size(Kxx,1);
    # n = size(Kxy,2);
    # r = size(Kxz,2);

    u_zz = (1./(nz*(nz-1))) * np.sum(Kzznd)
    u_zx = np.sum(Kzx) / (nz*nx)
    u_zy = np.sum(Kzy) / (nz*ny)
    # u_xx=sum(sum(Kxxnd))*( 1/(m*(m-1)) );
    # u_xy=sum(sum(Kxy))/(m*n);
    # u_xz=sum(sum(Kxz))/(m*r);

    ct1 = 1./(nz*(nz-1)**2) * np.sum(np.dot(Kzznd,Kzznd))
    # ct1 = (1/(m*(m-1)*(m-1)))   * sum(sum(Kzznd*Kzznd));
    ct2 = u_zz**2
    # ct2 =  u_xx^2;
    ct3 = 1./(nz*(nz-1)*ny) * np.sum(np.dot(Kzznd,Kzy))
    # ct3 = (1/(m*(m-1)*r))       * sum(sum(Kzznd*Kxz));
    ct4 = u_zz * u_zy
    # ct4 =  u_xx*u_xz;
    ct5 = (1./(nz*(nz-1)*nx)) * np.sum(np.dot(Kzznd, Kzx))
    # ct5 = (1/(m*(m-1)*n))       * sum(sum(Kzznd*Kxy));
    ct6 = u_zz * u_zx
    # ct6 = u_xx*u_xy;
    ct7 = (1./(nx*nz*ny)) * np.sum(np.dot(Kzx.T, Kzy))
    # ct7 = (1/(n*m*r))           * sum(sum(Kzx'*Kxz));
    ct8 = u_zx * u_zy
    # ct8 = u_xy*u_xz;

    zeta_1 = (ct1-ct2)-(ct3-ct4)-(ct5-ct6)+(ct7-ct8)
    # zeta_1 = (ct1-ct2)-(ct3-ct4)-(ct5-ct6)+(ct7-ct8);
    cov = (4.0*(nz-2))/(nz*(nz-1)) * zeta_1
    # theCov = (4*(m-2))/(m*(m-1)) * zeta_1;

    return cov

def med_heuristic(models, ref, subsample=1000, seed=100):
    # subsample first
    n = ref.shape[0]
    assert subsample > 0
    sub_models = []
    with util.NumpySeedContext(seed=seed):
        ind = np.random.choice(n, min(subsample, n), replace=False)
        for i in range(len(models)):
            sub_models.append(models[i][ind,:])
        sub_ref = ref[ind,:]

    med_mz = np.zeros(len(sub_models))
    for i, model in enumerate(sub_models):
        sq_pdist_mz = util.dist_matrix(model, sub_ref)**2
        med_mz[i] = np.median(sq_pdist_mz)**0.5

    sigma2 = 0.5*np.mean(med_mz)**2
    return sigma2
