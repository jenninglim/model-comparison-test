import numpy as np
import logging

from scipy.stats import truncnorm

def selection(Z):
    """
        Characterising selecting the top K Models from vector Z as a linear
        combination.
        input
            Z      : "Feature vector" with a normal distribution.
            K      :  Number of selections.
        return
            ind_sel: Selected index.
            A,b    : The linear combination of the selection event Az < b.
    """
    N = np.shape(Z)[0]
    ## Sorted list of Z
    ind_sorted = np.argsort(Z)
    ## Pick top k
    ind_sel = ind_sorted[0]

    A = np.zeros((N-1,N))
    for i in range(N-1):
        A[i, ind_sorted[0]] = 1
        A[i, ind_sorted[i+1]] = -1
    b = np.zeros((N-1))
    assert np.sum(np.matmul(A,Z) > 0) ==0, "Assumption error"
    return ind_sel, A, b

def psi_inf(A,b,eta, mu, cov, z):
    """
        Returns the p-value of the truncated normal. The mean,
        variance, and truncated points [a,b] is determined by Lee et al 2016.

    """
    l_thres, u_thres= calculate_threshold(z, A, b, eta, cov)
    sigma2 = np.matmul(eta,np.matmul(cov,eta))
    scale = np.sqrt(sigma2)

    params = {"u_thres":u_thres,
              "l_thres":l_thres,
              "mean": np.matmul(eta,mu),
              "scale":scale,
              }

    ppf = lambda x: truncnorm_ppf(x,
                        l_thres,
                        u_thres,
                        loc=np.matmul(eta,mu),
                        scale=scale)

    sf = lambda x: truncnorm.sf(x, l_thres/scale, u_thres/scale, scale=scale)
    return ppf, sf

def calculate_threshold(z, A, b, eta, cov):
    """
        Calculates the respective threshold for the method PSI_Inf.
    """
    etaz = eta.dot(z)
    Az = A.dot(z)
    Sigma_eta = cov.dot(eta)
    deno = Sigma_eta.dot(eta)
    alpha = A.dot(Sigma_eta)/deno

    assert(np.shape(A)[0] == np.shape(alpha)[0])
    pos_alpha_ind = np.argwhere(alpha>0).flatten()
    neg_alpha_ind = np.argwhere(alpha<0).flatten()
    acc = (b - np.matmul(A,z))/alpha+np.matmul(eta,z)

    if (np.shape(neg_alpha_ind)[0] > 0):
        l_thres = np.max(acc[neg_alpha_ind])
    else:
        l_thres = -10.0**10
    if (np.shape(pos_alpha_ind)[0] > 0):
        u_thres = np.min(acc[pos_alpha_ind])
    else:
        u_thres= 10**10
    return l_thres, u_thres

def test_significance(A, b, eta, mu, cov, z, alpha):
    """
        Compute an p-value by testing a one-tail.
        Look at right tail or left tail?
        Returns "h_0 Reject
    """
    ppf, sf = psi_inf(A, b, eta, mu, cov, z)
    stat = np.matmul(eta,z) ## Test statistic

    sigma = np.sqrt(np.matmul(eta,np.matmul(cov,eta)))

    ## If the std dev is < 0 or undefined, do not reject the hypothesis.
    if np.isnan(sigma) or not np.isreal(sigma):
        logging.warning("Scale is not real or negative, test reject")
        return False, 1.

    threshold = ppf(1.-alpha)
    pval = sf(stat)
    return stat > threshold, pval

def generateEta(ind_sel, n_models):
    """
        Generate multiple etas corresponding to testing
        within the selected indices.
    """
    etas = np.zeros((n_models-1, n_models))
    for i in range(n_models-1):
        index = i if i < ind_sel else i +1
        etas[i,ind_sel] = -1
        etas[i,index]=1
    return etas

def truncnorm_ppf(x, a, b,loc=0., scale=1.):
    """
        Approximate Percentile function of the truncated normal. Particularly in
        the tail regions (where the standard SciPy function may be undefined.
    """

    thres = truncnorm.ppf(x,(a-loc)/scale,(b-loc)/scale,loc=loc, scale=scale)

    if np.any(np.isnan(thres)) or np.any(np.isinf(thres)):
        logging.info("Threshold is Nan using approximations.")
        thres = loc+scale*quantile_tn(x,(a-loc)/scale,(b-loc)/scale)
    return thres

def quantile_tn(u,a,b,threshold=0.0005):
    """
        Approximate quantile function in the tail region
        https://www.iro.umontreal.ca/~lecuyer/myftp/papers/truncated-normal-book-chapter.pdf
    """

    def q(x, r=10):
        """
            Helper function.
        """
        acc=0

        for i in range(r):
            acc = acc + (2*i-1)/((-1)**i*x**(2*i+1))
        return 1/x + acc

    q_a = q(a)
    q_b = q(b)
    c =q_a * (1- u) + q_b * u * np.exp((a**2 - b**2)/2)
    d_x = 100
    z = 1 - u + u *  np.exp((a**2 - b**2)/2)
    x = np.sqrt(a**2 - 2 * np.log(z))

    while d_x > threshold and not np.isnan(d_x):
        z = z - x * (z * q(x) - c)
        x_new = np.sqrt(a**2 - 2 * np.log(z))
        d_x = np.abs(x_new - x)/x
        x = x_new
    return x

