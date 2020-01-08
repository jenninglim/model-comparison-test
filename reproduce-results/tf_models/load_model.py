import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import matplotlib.pyplot as plt

from reltest.density import from_tensorflow_to_UD
from .flow import OneGaussMAF, FivGaussMAF
from kgof import glo
from .config import CONFIG

DIR = CONFIG.DIR_TF_CKPTS
c_type = CONFIG.CRIME_TYPE
SHAPE_DIR = CONFIG.SHAPE_DIR
SHALE_FILE = "PVS_18_v2_unsd_17"

def load_flow_model(n_layer, n_gauss=1):
    tf.reset_default_graph()
    sess = tf.Session()
    name="{0}_{1}".format(n_layer, n_gauss)
    if n_gauss==5:
        model = FivGaussMAF(n_layer, name=name,use_batchnorm=False)
    elif n_gauss==1:
        model = OneGaussMAF(n_layer, name=name,use_batchnorm=False)
    else:
        assert(1==0)
    SAVE_DIR =DIR+"/{0}_{1}".format(n_layer, n_gauss)
    opt = tf.train.AdamOptimizer(1e-4).minimize(model.loss([[0.1,0.1]]))
    sess.run([tf.global_variables_initializer(),
            tf.local_variables_initializer()])
    saver = tf.train.Saver(tf.trainable_variables(model.name))
    max_index =   max([int(i) for i in os.listdir(SAVE_DIR)])
    saver.restore(sess, SAVE_DIR+"/{0}/model".format(max_index))
    return model, sess

def get_log_density(n_layer, n_gauss):
    model, sess = load_flow_model(n_layer, n_gauss)
    x = tf.placeholder(tf.float32, [None, 2], name="input")
    log_prob = model.log_prob(x)
    dlog = tf.gradients(log_prob,x)
    sess.graph.finalize()
    return log_prob, dlog, sess


def plot_density(UnormalisedDensities,x_lin,y_lin, axs, with_map=False):
    n_densities= len(UnormalisedDensities)
    train, test, mu, scale =load_crime_dataset('ROBBERY',7000,return_transform=True)
    mu, scale = mu.astype(np.float32), scale.astype(np.float32)
    train  = train*scale + mu
    X,Y = np.meshgrid(x_lin,y_lin)
    mat = np.dstack([X,Y])
    mesh_shape = mat.shape
    #mat = mat.reshape([-1,2])
    assert(len(axs) == n_densities)
    # MKDIR
    for i, UnormalisedDensity in enumerate(UnormalisedDensities):
        eval_density=np.exp(np.nan_to_num(UnormalisedDensity.log_den(mat)))
        #eval_density=eval_density.reshape([mesh_shape[0],mesh_shape[1]])
        if not with_map:
            log_den = UnormalisedDensity.log_den
            axs[i].contourf(X, Y,eval_density, levels=10)
            axs[i].set_xlim([np.min(x_lin),np.max(x_lin)])
            axs[i].set_ylim([np.min(y_lin),np.max(y_lin)])
        else: # WITH MAP
            m = Basemap(projection='lcc', resolution='h',
                lat_0=np.mean(train,axis=0)[1], lon_0=np.mean(train,axis=0)[0],
                width=5.E4, height=5.E4, ax=axs[i])
            m.readshapefile(SHAPE_DIR+ SHALE_FILE, 'shape')
            #
            m.contourf(scale[0]*X+mu[0], scale[1]*Y+mu[1],eval_density, levels=10,latlon=True,cmap=plt.get_cmap("YlOrRd"))
            m.readshapefile(SHAPE_DIR+ SHALE_FILE, 'shape')
            # axs[i].set_ylim(scale[0]*np.array([np.min(x_lin),np.max(x_lin)])+mu[0])
            # axs[i].set_xlim(scale[1]*np.array([np.min(y_lin),np.max(y_lin)])+mu[1])
    return axs

def plot_samples_map(ax):
    from mpl_toolkits.basemap import Basemap
    train, test, loc, scale= load_crime_dataset(c_type,11000, return_transform=True)
    samples = train * scale + loc
    m = Basemap(projection='lcc', resolution='h',
        lat_0=np.mean(samples,axis=0)[1], lon_0=np.mean(samples,axis=0)[0],
        width=5.E4, height=5.E4, ax=ax)
    counts,xbins,ybins = np.histogram2d(samples[:,0], samples[:,1], bins=100, range=[[m.llcrnrlon,m.urcrnrlon],[m.llcrnrlat,m.urcrnrlat]],density=True)
    m.imshow(counts.T, extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],alpha=1,cmap=plt.get_cmap("YlOrRd"))
    m.readshapefile(SHAPE_DIR+ SHALE_FILE, 'shape')
    return ax
 
def plot_density_hist(UnormalisedDensities,samples, axs,label=None):
    log_dens_samples=[UnormalisedDensity.log_den(samples) for UnormalisedDensity in UnormalisedDensities]
    # MKDIR
    for i, log_dens_sample in enumerate(log_dens_samples):
        axs[i].axvline(np.mean(log_dens_sample),label=label)
        axs[i].hist(log_dens_sample,bins=20, alpha =0.8, label=label,density=True)
        axs[i].set_xlim([-6,0])
    return axs

def plot_data():
    loc, _, shift, scale = load_crime_dataset(c_type,1000,True)
    plt.scatter(loc[:,0],loc[:,1])

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
