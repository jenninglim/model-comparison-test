import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import re

tfb = tfp.bijectors
tfd = tfp.distributions
### INVERTIBLE FLOW MODELS

class OneGaussMAF():
    def __init__(self, n_layers, use_batchnorm=False, name="default"):
        base_dist=tfd.MultivariateNormalDiag(
            loc=[0.,0.])

        with tf.variable_scope(name, dtype=tf.float32):
            self.model = tfd.TransformedDistribution(
                distribution= base_dist,
                bijector= AffineNonlinear(n_layers,name, use_batchnorm)
            )
        self.name = name
        self.n_layers =n_layers

    def loss(self, x):
        return -tf.reduce_mean(self.model.log_prob(x))

    def sample(self, n):
        return self.model.sample(n)

    '''
    def ploss(self, x, beta):
        loss = -tf.reduce_mean(self.model.log_prob(x))
        weights = []
        for var in tf.trainable_variables(self.name):
            if re.match("{0}_1/dense/kernel:*".format(self.name), var.name):
                weights.append(var)
        penalty = tf.reduce_sum([tf.nn.l2_loss(weight) for weight in weights])
        return loss +  beta*penalty
        '''
    def log_prob(self,x):
        return self.model.log_prob(x)

class FivGaussMAF():
    def __init__(self, n_layers, use_batchnorm=False, name="default"):
        n_gauss = 5.
        base_dist=tfd.Mixture(
          cat=tfd.Categorical(probs=[0.2,0.2,0.2,0.2,0.2]),
            components=[
            tfd.MultivariateNormalDiag(loc=[1., -1.]),
            tfd.MultivariateNormalDiag(loc=[-1., 1.]),
            tfd.MultivariateNormalDiag(loc=[0., 0.]),
            tfd.MultivariateNormalDiag(loc=[1., 1.]),
            tfd.MultivariateNormalDiag(loc=[-1., -1.]),
        ])

        with tf.variable_scope(name, dtype=tf.float32):
            self.model = tfd.TransformedDistribution(
                distribution= base_dist,
                bijector= AffineNonlinear(n_layers,name, use_batchnorm)
            )
        self.name = name
        self.n_layers =n_layers

    def loss(self, x):
        return -tf.reduce_mean(self.model.log_prob(x))

    '''
    def ploss(self, x, beta):
        loss = -tf.reduce_mean(self.model.log_prob(x))
        weights = []
        for var in tf.trainable_variables(self.name):
            if re.match("{0}_1/dense/kernel:*".format(self.name), var.name):
                weights.append(var)
        penalty = tf.reduce_sum([(weight**2) for weight in weights])
        return loss +  beta*penalty
        '''

    def sample(self, n):
        return self.model.sample(n)

    def log_prob(self,x):
        return self.model.log_prob(x)
def AffineNonlinear(n_layers,name,use_batchnorm):
    # Init variables
    layers = []
    for i in range(n_layers):
        layers.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                name=name,
                activation=tf.nn.leaky_relu,
                hidden_layers=[512, 512])))
        if use_batchnorm and i % 2 == 0:
            layers.append(tfb.BatchNormalization())
        layers.append(tfb.Permute(permutation=[1, 0]))
    return tfb.Chain(layers[:-1])
