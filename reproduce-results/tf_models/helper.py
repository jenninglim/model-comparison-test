import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from scipy import linalg
from .config import CONFIG

import reltest.util as util 

from kgof import glo 

def load_crime_dataset(c_type, size, return_transform=False):
    ## Take in consideration the mean and std
    import os
    dataset_dir = CONFIG.CRIME_DATASET_DIR
    dd = np.load(glo.data_file(dataset_dir))['data']
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
