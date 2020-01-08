import numpy as np

import reltest.mmd as mmd
import reltest.ksd as ksd

from reltest.mctest import MCTestPSI, MCTestCorr
from reltest import kernel
from kmod.mctest import SC_MMD 

def two_model_rej(source, n_samples, n_trials, eta, offset=0):
    """
        Rejection rate of PSIMMD_Bloc, PSIMMD_Inc, RelMMD for a given
        number of n_samples averaging over n_trials.
    """

    p_PSI_mmd_lin = np.zeros(n_trials)
    p_PSI_mmd_u   = np.zeros(n_trials)
    p_PSI_ksd_lin = np.zeros(n_trials)
    p_PSI_ksd_u = np.zeros(n_trials)
    p_mmd_rel = np.zeros(n_trials)
    p_ksd_rel = np.zeros(n_trials)
    model_densities = source.get_densities()

    ## Average P-Value over difference seeds
    for i in range(n_trials):
        # Sample Data
        P_0, P_1, Q = source.sample(n_samples,seed=i+n_trials*offset)
        samples = [P_0.data(), P_1.data()]

        # Determine parameters
        mmd_med = mmd.med_heuristic([P_0.data(), P_1.data()], Q.data(), subsample=1000, seed=i+3)
        ksd_med = ksd.med_heuristic(Q.data(), subsample=1000)

        # Initialise various MMD estimators
        mmd_u = mmd.MMD_U(kernel.KGauss(mmd_med))
        mmd_lin = mmd.MMD_Linear(kernel.KGauss(mmd_med))

        # Initiliaise KSD estimators
        ksd_u = ksd.KSD_U(kernel.KGauss(ksd_med))
        ksd_l = ksd.KSD_Linear(kernel.KGauss(ksd_med))

        psi_test = MCTestPSI(Q.data())
        cor_test = MCTestCorr(Q.data())

        mctest = lambda dist, models : psi_test.perform_test(models,
                dist,
                eta
                )['h0_rejected']

        ctest = lambda dist, models : cor_test.perform_test(models,
                dist,
                eta
                )['h0_rejected']

        # Calculate p_values PSI MMD
        p_PSI_mmd_lin[i] = mctest(mmd_lin, samples)
        p_PSI_mmd_u[i] = mctest(mmd_u, samples)

        # Calculate p_values PSI KSD
        p_PSI_ksd_u[i] = mctest(ksd_u,model_densities)
        p_PSI_ksd_lin[i] = mctest(ksd_l,model_densities)

        # Calculate mmd ksd
        p_mmd_rel[i] = ctest(mmd_u,samples)
        p_ksd_rel[i] = ctest(ksd_u,model_densities)

    rej_rate_PSI_mmd_lin = np.sum(p_PSI_mmd_lin)/n_trials
    rej_rate_PSI_mmd_u = np.sum(p_PSI_mmd_u)/n_trials
    rej_rate_PSI_ksd_lin = np.sum(p_PSI_ksd_lin)/n_trials
    rej_rate_PSI_ksd_u = np.sum(p_PSI_ksd_u)/n_trials
    rej_rate_rel_mmd = np.sum(p_mmd_rel)/n_trials
    rej_rate_rel_ksd = np.sum(p_ksd_rel)/n_trials

    results = {'PSI_mmd_lin':rej_rate_PSI_mmd_lin,
               'PSI_mmd_u':rej_rate_PSI_mmd_u,
               'PSI_ksd_u':rej_rate_PSI_ksd_u,
               'PSI_ksd_lin':rej_rate_PSI_ksd_lin,
               'RelMMD' :rej_rate_rel_mmd,
               'RelKSD' :rej_rate_rel_ksd}
    return results
