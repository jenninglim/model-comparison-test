import numpy as np
import matplotlib.pyplot as plt
import reltest
from reltest.mctest import MCTestPSI, MCTestCorr
import reltest.mmd as mmd
import reltest.ksd as ksd
from reltest import kernel
from kmod.mctest import SC_MMD
from freqopttest.util import meddistance
import logging
import sys
import os
from ex_models import generatelRBM

## Setting
n_samples = 1000
ydim = 20
xdim = 5
n_trials = 300
to_perturb = float(sys.argv[1])
perturbs = [0.2, 0.25, 0.35, 0.4, 0.45, 0.5]
n_models = len(perturbs)
perturbs[1] = to_perturb
print("perturb " + str(to_perturb))
print(perturbs)

def independent_test(perturbs, n_samples, n_trials, setting):
    src = generatelRBM(perturbs, ydim, xdim)
    res_psi=  {'mmd_u':[],
               'mmd_lin':[],
               'ksd_u':[],
               'ksd_lin':[],
              }
    res_cor= {
              #'ksd_u_bh':[],
              'ksd_u_by':[],
             # 'ksd_u_bn':[],
              #'mmd_u_bh':[],
              'mmd_u_by':[],
              #'mmd_u_bn':[],
              }
    model_dens = src.get_densities()
    for j in range(n_trials):
        models, Q = src.sample(n_samples, seed=j)

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
        #res_cor['mmd_u_bh'].append(corrtest.perform_tests(model_samples, Q.data(), n_samples, mmd_u, split=0.5, density=False, correction=0))
        res_cor['mmd_u_by'].append(corrtest.perform_tests(model_samples, mmd_u, split=0.5, density=False, correction=1))
#         res_cor['mmd_u_bn'].append(corrtest.perform_tests(model_samples, Q.data(), n_samples, mmd_u, split=True, density=False, correction=2))
#        res_cor['ksd_u_bh'].append(corrtest.perform_tests(model_dens, Q.data(), n_samples, ksd_u, split=0.5, density=True, correction=0))
        res_cor['ksd_u_by'].append(corrtest.perform_tests(model_dens, ksd_u, split=0.5, density=True, correction=1))
#         res_cor['ksd_u_bn'].append(corrtest.perform_tests(model_dens, Q.data(), n_samples, ksd_u, split=True, density=True, correction=2))
    return res_psi,res_cor

setting = {'n':n_models,
       'dim':ydim}
SAVE_DIR = "./temp/rbm/"

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
np.save(SAVE_DIR+"PSI"+str(to_perturb),0)
np.save(SAVE_DIR+"COR"+str(to_perturb),0)

res_psi,res_cor = independent_test(perturbs,n_samples,n_trials,setting)
np.save(SAVE_DIR+"PSI"+str(to_perturb),res_psi)
np.save(SAVE_DIR+"COR"+str(to_perturb),res_cor)
