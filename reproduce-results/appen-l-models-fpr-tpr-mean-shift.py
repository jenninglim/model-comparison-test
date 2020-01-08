import numpy as np
import reltest
from reltest.mctest import MCTestPSI, MCTestCorr
import reltest.mmd as mmd
import reltest.ksd as ksd
from reltest import kernel

import logging
from ex_models import generateLGauss
from helper import summary
import sys
import os

dim = 10
n_models = 10
n_same = 9

model_params = {'mu0':0.5, 'sig0':1, # Model 0 Parameters
                'muR':0, 'sigR':1  # Reference Parameters
                 }

n_samples = int(sys.argv[1])
src = generateLGauss(model_params, dim, n_models,n_same)
res = src.sample(n_samples)
n_trials =300
models = res['models']
Q = res['ref']

def independent_test(n_samples, n_trials, src, setting):

    res_psi=  {'mmd_u':[],
               'mmd_lin':[],
               'ksd_u':[],
               'ksd_lin':[],
              }
    res_cor= {
              'ksd_u_bh':[],
              'ksd_u_by':[],
              'ksd_u_bn':[],
              'mmd_u_bh':[],
              'mmd_u_by':[],
              'mmd_u_bn':[],
              }
    model_dens = src.get_densities()
    for j in range(n_trials):
        samples = src.sample(n_samples, seed=j)

        models = samples['models']
        Q = samples['ref']

        psiTest = MCTestPSI(Q.data())
        corrtest = MCTestCorr(Q.data())

        mmd_med = mmd.med_heuristic([i.data() for i in models], Q.data(),
            subsample=1000)
        ksd_med = ksd.med_heuristic(Q.data(),
            subsample=1000)

        mmd_kernel, ksd_kernel = kernel.KGauss(mmd_med), kernel.KGauss(ksd_med)
        mmd_u = mmd.MMD_U(mmd_kernel)
        mmd_lin = mmd.MMD_Linear(mmd_kernel)
        ksd_u = ksd.KSD_U(ksd_kernel)
        ksd_lin = ksd.KSD_Linear(ksd_kernel)
        model_samples = [i.data() for i in models]
        ## PSI Based Test
        res_psi['ksd_u'].append(psiTest.perform_tests(model_dens, ksd_u))
        res_psi['ksd_lin'].append(psiTest.perform_tests(model_dens, ksd_lin))
        res_psi['mmd_u'].append(psiTest.perform_tests(model_samples, mmd_u))
        res_psi['mmd_lin'].append(psiTest.perform_tests(model_samples, mmd_lin))
        ## Correction Based Test
        res_cor['mmd_u_bh'].append(corrtest.perform_tests(model_samples, mmd_u, split=0.5, density=False, correction=0))
        res_cor['mmd_u_by'].append(corrtest.perform_tests(model_samples, mmd_u, split=0.5, density=False, correction=1))
        res_cor['mmd_u_bn'].append(corrtest.perform_tests(model_samples, mmd_u, split=0.5, density=False, correction=3))
        res_cor['ksd_u_bh'].append(corrtest.perform_tests(model_dens, ksd_u, split=0.5, density=True, correction=0))
        res_cor['ksd_u_by'].append(corrtest.perform_tests(model_dens, ksd_u, split=0.5, density=True, correction=1))
        res_cor['ksd_u_bn'].append(corrtest.perform_tests(model_dens, ksd_u, split=0.5, density=True, correction=3))
    return res_psi,res_cor

setting = {'n':n_models,
       'dim':dim}

res_psi,res_cor = independent_test(n_samples,n_trials,src,setting)

np.save(SAVE_DIR+"PSI"+str(n_samples),res_psi)
np.save(SAVE_DIR+"COR"+str(n_samples),res_cor)
