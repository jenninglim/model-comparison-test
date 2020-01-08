import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from scipy import linalg

import reltest.util as util 
from reltest.mctest import MCTestPSI
from reltest.mmd import MMD_Linear, MMD_U
from reltest.ksd import KSD_U, KSD_Linear
from reltest import kernel

from kmod.mctest import SC_MMD
from kgof import glo 
from rej import *

def two_model_rej_samp(source, l_samples, n_trials, eta, n_selected =1):
    """
        Rejection rate of PSIMMD_Bloc, PSIMMD_Inc, RelMMD for a given
        a range of sample sizes determined by l_samples
    """
    res_psi_mmd_lin = np.zeros((len(l_samples),1))
    res_psi_mmd_inc = np.zeros((len(l_samples),1))
    res_psi_mmd_bloc = np.zeros((len(l_samples),1))
    res_psi_mmd_u = np.zeros((len(l_samples),1))
    res_psi_ksd_u = np.zeros((len(l_samples),1))
    res_psi_ksd_lin = np.zeros((len(l_samples),1))
    res_rel_mmd = np.zeros((len(l_samples),1))
    res_rel_ksd = np.zeros((len(l_samples),1))

    ## Average P-Value over difference seed
    for j in range(len(l_samples)):
        logging.info("Testing for %d samples" % l_samples[j])
        n_samples = l_samples[j]
        block_size = int(np.sqrt(n_samples))

        one_res = two_model_rej(source, n_samples, n_trials, eta, offset=j)

        res_psi_mmd_lin[j] = one_res['PSI_mmd_lin']
        res_psi_mmd_u[j] = one_res['PSI_mmd_u']
        res_psi_ksd_u[j] = one_res['PSI_ksd_u']
        res_psi_ksd_lin[j] = one_res['PSI_ksd_lin']
        res_rel_mmd[j] = one_res['RelMMD']
        res_rel_ksd[j] = one_res['RelKSD']

    results = {
               'PSI_mmd_lin':res_psi_mmd_lin,
               'PSI_mmd_u':res_psi_mmd_u,
               'PSI_ksd_lin':res_psi_ksd_lin,
               'PSI_ksd_u':res_psi_ksd_u,
               'RelMMD' :res_rel_mmd,
               'RelKSD' :res_rel_ksd}
    return results

def neg_log_likelihood(log_ds, samples):
    return [-np.mean(log_d(samples)) for log_d in log_ds]

def filter_crimetype(data, type = None):
    if type is None:
        data = data
    else:
        data = data[data[:,0] == type]
    if len(data) == 1:
        print("No Crime Type found")
    else:
        loc = data[:,1:].astype(float)
        loc = np.nan_to_num(loc)
        loc = loc[loc[:,0] != 0]
        #Set City bound
        loc = loc[loc[:,0] >-89]
        loc = loc[loc[:,1] > 40]
        return loc

def load_crime_dataset(c_type, size, return_transform=False):
    ## Take in consideration the mean and std
    import os
    dd = np.load(glo.data_file('/is/ei/jlim/Documents/n-relative-testing/data/chicago_crime_loc_with_type2016.npz'))['data']
    loc = filter_crimetype(dd, c_type)
    ## Standardise
    shift, scale = np.mean(loc,axis=0), np.std(loc,axis=0)
    loc = loc - shift
    loc = loc/scale
    loc_train, loc_test = loc[:size,:], loc[size:,:]
    def init(loc_test):
        def sample_test_data(size, seed):
            with util.NumpySeedContext(seed=seed):
                sample_test = np.random.permutation(loc_test)
            return sample_test[:size,:]
        return sample_test_data
    if return_transform:
        return loc_train,init(loc_test), shift, scale
    else:
        return loc_train,init(loc_test)

def summary(results, n_models):
    """
    Return Summary of results:
        Average Selection:
        Average Rejection:
        Time:
    """
    av_rej = np.zeros(n_models)
    av_sel = np.zeros(n_models)
    av_time = 0
    for result in results:
        av_rej = av_rej+result['h0_rejected']/len(results)
        av_sel[result['ind_sel']] += 1./len(results)
        av_time = av_time+result['time_secs']/len(results)
    summary = {'av_rej': av_rej,
               'av_sel':av_sel,
               'av_time':av_time}
    return summary

def download_to(url, file_path):
    """
    Download the file specified by the URL and save it to the file specified
    by the file_path. Overwrite the file if exist.
    """

    # see https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    import urllib.request
    import shutil
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, \
            open(file_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

########################
#based on https://github.com/mbinkowski/MMD-GAN
#and https://github.com/wittawatj/kernel-mod/blob/master/kmod/ex/exutil.py
###############################

def fid_score(codes_g, codes_r, eps=1e-6, output=sys.stdout, **split_args):
    splits_g = get_splits(**split_args)
    splits_r = get_splits(**split_args)
    assert len(splits_g) == len(splits_r)
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d

    scores = np.zeros(len(splits_g))
    for i, (w_g, w_r) in enumerate(zip(splits_g, splits_r)):
        part_g = codes_g[w_g]
        part_r = codes_r[w_r]

        mn_g = part_g.mean(axis=0)
        mn_r = part_r.mean(axis=0)

        cov_g = np.cov(part_g, rowvar=False)
        cov_r = np.cov(part_r, rowvar=False)

        covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
        if not np.isfinite(covmean).all():
            cov_g[range(d), range(d)] += eps
            cov_r[range(d), range(d)] += eps
            covmean = linalg.sqrtm(cov_g.dot(cov_r))

        scores[i] = np.sum((mn_g - mn_r) ** 2) + (
            np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
    return np.real(scores)

def get_splits(n, splits=10, split_method='openai'):
    if split_method == 'openai':
        return [slice(i * n // splits, (i + 1) * n // splits)
                for i in range(splits)]
    elif split_method == 'bootstrap':
        return [np.random.choice(n, n) for _ in range(splits)]
    elif 'copy':
        return [np.arange(n) for _ in range(splits)]
    else:
        raise ValueError("bad split_method {}".format(split_method))

def fid(X, Z):
    """
    Compute the FIDs FID(P, R) and FIR(Q, R).
    The bootstrap estimator from Binkowski et al. 2018 is used.
    The number of bootstrap sampling can be specified by the variable splits
    below. For the method for the non-bootstrap version, see the method
    met_fid_nbstrp.
    """

    # keeping it the same as the comparison in MMD gan paper, 10 boostrap resamplings
    splits = 10
    split_size = X.shape[0]
    assert X.shape == Z.shape
    split_method = 'bootstrap'
    split_args = {'splits': splits, 'n': split_size, 'split_method': split_method}

    with util.ContextTimer() as t:
        fid_scores_xz = fid_score(X, Z, **split_args)

        fid_score_xz = np.mean(fid_scores_xz)

    return fid_score_xz
