#!/home/jlim/anaconda3/envs/test/bin/python
import os
import numpy as np

from tf_models.load_model import get_log_density, plot_density,plot_density_hist,plot_samples_map
from tf_models.mixture import GMM_Fit

from reltest.mctest import MCTestPSI, MCTestCorr
from tf_models.helper import load_crime_dataset
from reltest.ksd import KSD_U, KSD_Linear, med_heuristic
from reltest.kernel import KGauss, KIMQ
from reltest import density
from reltest.density import from_tensorflow_to_UD

import matplotlib.pyplot as plt
import tensorflow as tf

c_type = 'ROBBERY'
SIZE = 7000
n_components = [1,2,5]
train, test = load_crime_dataset(c_type,SIZE)
gmm = GMM_Fit(n_components, train)
del train
models_layer = [1,5]
n_gauss = [1,5]
flow_den = [from_tensorflow_to_UD(2,*get_log_density(*params)) for params in zip(models_layer, n_gauss)]
n_nf = len(flow_den)
n_gmm = len(gmm.get_densities())
n_models = n_gmm+n_nf

candidate_models = gmm.get_densities() + flow_den
n_models = len(candidate_models)

SAVE_DIR="./temp/gk/"
n_trials=100
for i in range(n_trials):
    test_set = test(2000, i)
    mctest = MCTestPSI(test_set)
    corrtest = MCTestCorr(test_set)
    med = med_heuristic(test_set)
    ksd_u = KSD_U(KGauss(med))
    test_set = test_set.astype(np.float32)
    mctest_res = mctest.perform_tests(candidate_models, ksd_u)
    ctest_res = corrtest.perform_tests(candidate_models, ksd_u, split=0.5, density=True, correction=2)
    model_log_den = np.zeros(n_models)
    for j, candidate in enumerate(candidate_models):
        model_log_den[j] = np.mean(candidate.log_den(test_set))
    np.save(SAVE_DIR+"cor_res_{0}.npy".format(i),ctest_res)
    np.save(SAVE_DIR+"log_den_{0}.npy".format(i),model_log_den)
    np.save(SAVE_DIR+"psi_res_{0}.npy".format(i),mctest_res)
