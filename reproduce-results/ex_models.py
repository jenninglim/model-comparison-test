import autograd.numpy as np
from future.utils import with_metaclass
from kgof.data import Data
from abc import ABCMeta, abstractmethod
import scipy.stats as stats
import autograd.scipy.stats as diff_stats
import kgof.density as density
import kmod.model as model

"""
Some example models for toy experiments.
    TODO: Wrap get_densities() with from_log_den.
"""

class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)

# end NumpySeedContext
class ToyProblem(with_metaclass(ABCMeta, object)):
    def n_sources(self):
        raise NotImplementedError()    

    def get_densities(self):
        raise NotImplementedError()    
    
    def sample(self, n_samples, seed):
        raise NotImplementedError()    

 
class generateTwoGauss(ToyProblem):
    """
        Data Source for Generating Two Gaussian
        Candidate Models and a "Reference" distribution.
    """
    def __init__(self, params, dim, n_models):
        self.d = dim
        self.params = params

        def mu(x):
            mu = np.zeros((dim))
            mu[0] = x
            return mu

        mu0,cov0, = mu(self.params['mu0']), self.params['sig0']*np.eye(dim)
        mu1,cov1, = mu(self.params['mu1']), self.params['sig1']*np.eye(dim)
        muR,covR, = mu(self.params['muR']), self.params['sigR']*np.eye(dim)

        self.p0 = stats.multivariate_normal(mu0, cov0)
        self.p1 = stats.multivariate_normal(mu1, cov1)
        self.q  = stats.multivariate_normal(muR, covR)

        self.n_sources = n_models

    def sample(self, n, seed=3):
        dim = self.d
        def mu(x):
            mu = np.zeros((dim))
            mu[0] = x
            return mu

        mu0,cov0, = mu(self.params['mu0']), self.params['sig0']*np.eye(dim)
        mu1,cov1, = mu(self.params['mu1']), self.params['sig1']*np.eye(dim)
        muR,covR, = mu(self.params['muR']), self.params['sigR']*np.eye(dim)

        with NumpySeedContext(seed=seed):
            self.p0 = stats.multivariate_normal(mu0, cov0)
            self.p1 = stats.multivariate_normal(mu1, cov1)
            self.q  = stats.multivariate_normal(muR, covR)
            X = self.p0.rvs(size=n)
            Y = self.p1.rvs(size=n)
            Q = self.q.rvs(size=n)

        if X.ndim == 1:
            X = np.expand_dims(X,axis=1)
        if Y.ndim == 1:
            Y = np.expand_dims(Y,axis=1)
        if Q.ndim == 1:
            Q = np.expand_dims(Q,axis=1)
        return Data(X), \
               Data(Y), \
               Data(Q)

    def get_densities(self):
        log_p0 = lambda x: diff_stats.multivariate_normal.logpdf(x,mean=self.p0.mean,cov=self.p0.cov)
        log_p1 = lambda x: diff_stats.multivariate_normal.logpdf(x,mean=self.p1.mean,cov=self.p1.cov)
        return [density.from_log_den(self.d,log_p0), density.from_log_den(self.d,log_p1)]

class generateLGauss(ToyProblem):
    def __init__(self, params, dim, n_models, n_same):
        self.d = dim
        mu0,sig0, = params['mu0'], params['sig0']
        muR,sigR, = params['muR'], params['sigR']

        mean = np.zeros((n_models, dim))
        for i in range(n_same):
            sign = 1 if i % 2==0 else -1
            mean[i,int(np.floor(i/2))] = sign* mu0
        for i in range(n_same,n_models):
            sign = 1 if i % 2==0 else -1
            mean[i,int(np.floor((i-n_same)/2))%dim] = (1.+0.2*np.floor((i-n_same)/2)) * sign
        self.models = []
        for i in range(n_models):
            self.models = self.models + [stats.multivariate_normal(mean[i,:], sig0*np.eye(dim))]

        meanR = np.zeros(dim)
        meanR[0] = muR
        self.q  = stats.multivariate_normal(meanR, sigR*np.eye(dim))

    def sample(self, n, seed=3):
        with NumpySeedContext(seed=seed):
            model_samples = [i.rvs(size=n) for i in self.models]
            Q = self.q.rvs(size=n)
            ## Expand dims
            model_samples = [np.expand_dims(i, axis=1) if i.ndim == 1 else i for i in model_samples]
            if Q.ndim == 1:
                Q = np.expand_dims(Q,axis=1)
            res = {'models':[Data(i) for i in model_samples], 
                    'ref': Data(Q)}
            return res

    def get_densities(self):
        def log_d(mean, cov):
            return density.from_log_den(self.d,lambda x: diff_stats.multivariate_normal.logpdf(x,mean=mean,cov=cov))
        return [log_d(model.mean, model.cov) for model in self.models]

class generate2dGauss(ToyProblem):
    def __init__(self):
        q_mean = np.array([0,0])
        means = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        multi = np.array([0.1,0.2,0.3,0.4])+1
        means_o = ([means[i]*multi[i] for i in range(4)])
        p_means = np.array(np.vstack((means, means_o)))
        q_cov = np.diag([1,1]) * 1
        n_models = 8

        self.models = [stats.multivariate_normal(mean, q_cov) for mean in p_means]
        self.q  = stats.multivariate_normal(q_mean, q_cov)

    def sample(self, n, seed=3):
        with NumpySeedContext(seed=seed):
            model_samples = [i.rvs(size=n) for i in self.models]
            Q = self.q.rvs(size=n)
            ## Expand dims
            model_samples = [np.expand_dims(i, axis=1) if i.ndim == 1 else i for i in model_samples]
            if Q.ndim == 1:
                Q = np.expand_dims(Q,axis=1)
            res = {'models':[Data(i) for i in model_samples], 
                    'ref': Data(Q)}
            return res

    def get_densities(self):
        return [model.logpdf for model in self.models]

class generateMultMod(ToyProblem):
    def __init__(self):
        self.means = np.array([[-1.0, 1], [1, 1], [-1, -1], [1, -1]])*3.5
        base_cov = np.array([[5.0, 0], [0, 0.5]])
        self.covr = np.tile(base_cov, [4, 1, 1])
        self.covq = np.tile(rot2d_cov(np.pi/5.0, base_cov), [4, 1, 1])
        self.covp = np.tile(rot2d_cov(np.pi/2.0, base_cov), [4, 1, 1])
    
    def sample(self, m,seed):
        with NumpySeedContext(seed=seed):
            p = density.GaussianMixture(self.means, self.covp)
            q = density.GaussianMixture(self.means, self.covq)
            r = density.GaussianMixture(self.means, self.covr)
            dsp, dsq, dsr = [P.get_datasource() for P in [p, q, r]]
            datp, datq, datr = [ds.sample(m, seed) for ds in [dsp, dsq, dsr]]
        return datp,datq,datr
    
    def get_densities(self):
        p = density.GaussianMixture(self.means, self.covp)
        q = density.GaussianMixture(self.means, self.covq)
        return [p,q]

def rot2d_matrix(angle):
    import math
    r = np.array( [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]] )
    return r

def rot2d_cov(angle, cov):
    R = rot2d_matrix(angle)
    return np.dot(np.dot(R, cov), R.T)

class generateRBM(ToyProblem):
    def __init__(self, to_perturb_Bp, to_perturb_Bq, dy, dx):
        """
            Perturbing
        """
        with NumpySeedContext(seed=11):
            B = np.random.randint(0, 2, (dy, dx))*2 - 1.0
            b = np.random.randn(dy)
            c = np.random.randn(dx)
            r = density.GaussBernRBM(B, b, c)

            # for p
            Bp_perturb = np.copy(B)
            Bp_perturb[0, 0] = Bp_perturb[0, 0] + to_perturb_Bp

            # for q
            Bq_perturb = np.copy(B)
            Bq_perturb[0, 0] = Bq_perturb[0, 0] + to_perturb_Bq

            p = density.GaussBernRBM(Bp_perturb, b, c)
            q = density.GaussBernRBM(Bq_perturb, b, c)
            self.dq = r.get_datasource(burnin=2000)
            self.p = (model.ComposedModel(p=p))
            self.q = (model.ComposedModel(p=q))

    def sample(self, m,seed):
        with NumpySeedContext(seed=seed):
            datp = self.p.get_datasource().sample(m,seed)
            datq = self.q.get_datasource().sample(m,seed)
            datr = self.dq.sample(m,seed)
        return datp,datq,datr
    
    def get_densities(self):
        return [self.p.get_unnormalized_density(), self.q.get_unnormalized_density()]

class generatelRBM(ToyProblem):
    def __init__(self, to_perturb_ms, dy, dx):
        """
            Perturbing
        """
        with NumpySeedContext(seed=11):
            B = np.random.randint(0, 2, (dy, dx))*2 - 1.0
            b = np.random.randn(dy)
            c = np.random.randn(dx)
            r = density.GaussBernRBM(B, b, c)

            model_Bs = []
            # for p
            for perturb in to_perturb_ms:
                B_perturb = np.copy(B)
                B_perturb[0,0] = B_perturb[0,0] + perturb
                model_Bs = model_Bs + [B_perturb]

            models_den = [density.GaussBernRBM(B_perturb_m, b, c) for B_perturb_m in model_Bs]

            self.dr = r.get_datasource(burnin=2000)
            self.models = [model.ComposedModel(p=model_den) for model_den in models_den]

    def sample(self, m,seed):
        with NumpySeedContext(seed=seed):
            datr = self.dr.sample(m,seed)
            m_samples =[model.get_datasource().sample(m,seed) for model in self.models]
        return m_samples, datr
    
    def get_densities(self):
        return [model.get_unnormalized_density() for model in self.models]

