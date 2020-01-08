import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import tensorflow as tf
import tensorflow_probability as tfp
from reltest.density import from_tensorflow_to_UD
from reltest import util
tfd = tfp.distributions

class GMM_Fit():
    def __init__(self, n_components, train):
        # models
        models = [mixture.GaussianMixture(n_components=n_comp, covariance_type='full') for n_comp in n_components]
        # train
        with util.NumpySeedContext(seed=2):
            models = [model.fit(train) for model in models]
        # Get Params
        muss, covss, weights = [model.means_.astype(np.float32) for model in models],[model.covariances_.astype(np.float32) for model in models],[model.weights_.astype(np.float32) for model in models]
        params = zip(n_components, weights, muss, covss)

        ## Tensorflow
        def gmm(n_components, mix, muss, covss):
            tf.reset_default_graph()
            components = []
            mus, covs = np.split(muss, n_components,axis=0), np.split(covss, n_components,axis=0)
            for mu, cov in zip(mus,covs):
                mvn = tfd.MultivariateNormalFullCovariance(loc=mu[0], covariance_matrix=cov[0])
                components.append(mvn)

            return tfd.Mixture(
              cat=tfd.Categorical(probs=mix),
              components=components,
            )

        def init_gmm_logp(gmm):
            sess = tf.Session()
            x = tf.placeholder(tf.float32, [None, 2], name="input")
            logp = gmm.log_prob(x)
            dlogp = tf.gradients(logp, x)
            sess.graph.finalize()
            return logp,dlogp,sess

        gmms = [init_gmm_logp(gmm(*param)) for param in params]
        ud_gmms = [from_tensorflow_to_UD(train.shape[1],*gmm)for gmm in gmms]
        self.ud_gmms = ud_gmms
    def get_densities(self):
        return self.ud_gmms
