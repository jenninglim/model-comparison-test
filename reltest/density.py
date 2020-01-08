###
# Functions for working with density functions implemented in tensorflow.
###
import tensorflow as tf
from kgof import density
import numpy as np

def from_tensorflow_to_UD(dim, log_prob, dlog_prob, sess):
    def log_den(y):
        with sess.graph.as_default():
            return sess.run(log_prob, feed_dict={"input:0":y})
    def dlog_den(y):
        with sess.graph.as_default():
            return sess.run(dlog_prob[0], feed_dict={"input:0":y})
    return from_log_den_and_grad(dim, log_den, dlog_den)

def from_log_den_and_grad(dim, log_den, dlog_den):
    return density.UDFromCallable(dim,flog_den=log_den, fgrad_log=dlog_den)
