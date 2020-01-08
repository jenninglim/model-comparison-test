import numpy as np

from future.utils import with_metaclass

from reltest import kernel
from reltest.psi import psi_inf, selection, test_significance, generateEta
from reltest import util

from abc import ABCMeta, abstractmethod

class PSITest(with_metaclass(ABCMeta, object)):
    def compute_params(self, models, ):
        """
            Compute Z, mu, sigma.
        """
        raise NotImplementedError()

    def perform_test(self, models, ref, n_samples, dist, eta):
        """
            Perform a single test for a given hypothesis determined by eta.
        """
        raise NotImplementedError()

    def split_samples(self, models, ref):
        """
            Split the data set.
        """
        raise NotImplementedError()

    def perform_tests(self, models, ref, n_samples, dist, n_selected):
        """
            Perform multiple testing depending on the selection event.
        """
        raise NotImplementedError()

class MCTestPSI(PSITest):
    def __init__(self, ref, alpha=0.05):
        self.alpha = alpha
        self.ref = ref


    def perform_test(self, models,  dist, eta):
        ref = self.ref
        l, dim, n_samples = len(models), ref.shape[1], ref.shape[0]

        with util.ContextTimer() as t:
            Z, mu,sigma = compute_params(models, ref, dist)
            ind_sel, A, b = selection(Z)
            stat = np.matmul(eta,Z)

            h0_rej, pval = test_significance(
                    A,
                    b,
                    eta,
                    np.zeros(l),
                    sigma,
                    Z,
                    self.alpha)
        results =  {
                    'test_stat': stat,
                    'sel_ind': ind_sel,
                    'pval': pval,
                    'h0_rejected': h0_rej,
                    'time_secs':t.secs}
        return results

    def perform_tests(self, models, dist):
        ref = self.ref
        l, dim, n_samples = len(models), ref.shape[1], ref.shape[0]

        with util.ContextTimer() as t:
            Z, mu,sigma = compute_params(models, ref, dist)
            ind_sel, A, b = selection(Z)
            etas = generateEta(ind_sel, l)
            h0_rejects = np.zeros(l, dtype=bool)
            pvals = np.ones(l)

            for i in range(l-1):
                eta = etas[i,:]
                ind = i if i < ind_sel else i +1

                h0_rejects[ind], pvals[ind] = test_significance(
                        A, ## Selection event parameter
                        b, ## Selection event parameter
                        eta, ## Hypothesis test parameter
                        np.zeros(l), ## Mean of the null distribution
                        sigma, ## Variance of the null distribution
                        Z,  ## Realisation of the test statistic
                        self.alpha)

        results = {'ind_sel': ind_sel,
                   'h0_rejected': h0_rejects,
                   'pvals':  pvals,
                   'time_secs':t.secs}
        return results

class MCTestCorr():
    """
        Baseline Comparision with correction
    """
    def __init__(self, ref, alpha=0.05):
        self.alpha = 0.05
        self.ref = ref

    def perform_test(self, models, dist, eta):
        """
            Testing for a single hypthothesis => no mutliple correction needed.
        """
        ref = self.ref
        l, dim, n_samples = len(models), ref.shape[1], ref.shape[0]

        with util.ContextTimer() as t:
            z, mu, sigma = compute_params(models, ref, dist)
            stat = np.dot(eta,z)
            pval = self.pvalue(stat,eta,sigma)
        results =  {
                    'h0_rejected': pval < self.alpha,
                    'time_secs':t.secs}
        return results

    def perform_tests(self, models, dist, split=1/2, correction=0, density=False):
        """
         Correction 0 with BH procedure
                    1 with BY procedure
        """
        ref = self.ref
        l, dim, n_samples = len(models), ref.shape[1], ref.shape[0]

        if split:
            if not density:
                models_sel = [model[:int(n_samples*split)] for model in models]
                models_tst = [model[int(n_samples*split):] for model in models]
            else:
                models_sel, models_tst = models,models
            ref_sel = ref[:int(n_samples*split)]
            ref_tst = ref[int(n_samples*split):]
        else:
            models_sel, models_tst = models,models
            ref_sel, ref_tst = ref, ref

        with util.ContextTimer() as t:
            score = compute_params(models_sel, ref_sel, dist, compute_mu_var=False)
            min_ind = np.argmin(score)
            tests = []
            for i in range(l):
                if i!= min_ind:
                    eta = np.zeros((l,1))
                    eta[min_ind,0] = -1
                    eta[i,0] = 1
                    tests.append(eta)

            pvalues = np.ones(l-1)
            z,mu,sigma2 = compute_params(models_tst, ref_tst, dist)

            for i,test in enumerate(tests):
                stat = np.matmul(test.T,z[:,np.newaxis])
                pvalues[i] = self.pvalue(stat,test,sigma2)

            if correction ==0:
                rej = self.bhcorrection(pvalues)
            elif correction ==1:
                rej = self.bycorrection(pvalues)
            elif correction ==2:
                rej = self.bocorrection(pvalues)
            elif correction ==3:
                rej = pvalues < self.alpha
            h0_rejected = np.zeros(l)
            for i in range(l-1):
                ind = i if i < min_ind else i + 1
                h0_rejected[ind] = rej[i]
        results =  {
                    'ind_sel': min_ind,
                    'h0_rejected': h0_rejected,
                    'time_secs':t.secs}
        return results

    def bocorrection(self, pvals):
        '''
        Bonferroni correction
        '''
        return pvals < self.alpha/(pvals.shape[0])

    def bhcorrection(self, pvals):
        '''
            benjamini–hochberg correction
        '''
        m = pvals.shape[0]
        l = m + 1
        k = np.zeros(l)
        ordered_pvals = np.sort(pvals)
        for i,pval in enumerate(ordered_pvals):
            if pval <= (i+1)/m*self.alpha:
                k[i]=1
        reverse = np.hstack([np.where(pvals.argsort()==i)[0] for i in range(len(pvals))])
        return k[reverse]

    def bycorrection(self, pvals):
        '''
            benjamini–Yekutieli correction
        '''
        m = pvals.shape[0]
        l = m + 1
        def c(m):
            return np.sum([1./(i+1) for i in range(m)])
        k = np.zeros(l)
        ordered_pvals = np.sort(pvals)
        for i,pval in enumerate(ordered_pvals):
            if pval <= (i+1)/m/c(m)*self.alpha:
                k[i]=1
        reverse = np.hstack([np.where(pvals.argsort()==i)[0] for i in range(len(pvals))])
        return k[reverse]
    
    def pvalue(self,stat, eta, cov):
        from scipy.stats import norm
        sigma2 = np.matmul(np.matmul(eta.T,cov),eta)
        if (sigma2 <=0):
            return 1.
        scale = np.sqrt(sigma2)
        return norm(loc=0.,scale=scale).sf(stat)
        

def compute_params(models, ref, dist, compute_mu_var=True):
    """
        Compute the parameters of the normal distribution 
        Z \sim N(mu, sigma).
    """
    l = len(models)
    n_samples = ref.shape[0]

    # Compute Z
    Z = np.zeros(l)
    for i in range(l):
        Z[i] = dist.compute(models[i], ref)

    Z = np.sqrt(dist.n_estimates(n_samples)) * Z

    # Compute variance and covariance
    if compute_mu_var:
        sigma2 = dist.compute_covariance(models, ref)
        sigma2 = sigma2 * dist.n_estimates(n_samples)
        mu = np.zeros(l)
        return Z, mu, sigma2
    return Z
