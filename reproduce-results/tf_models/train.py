#!/home/jlim/anaconda3/envs/test/bin/python
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .flow import OneGaussMAF, FivGaussMAF
import sys
from kgof import glo, util
from .load_model import load_crime_dataset
from .config import CONFIG

## TRAINING HYPERPARAMETERS

BATCH_SIZE = 7000
N_EPOCHS = int(2e6)
LEARNING_RATE = 1e-4
BETA = 1e-2

## DATASET PARAMETERS
NAME='CHICAGO'
DIM = 2
c_type = 'ROBBERY'
TRAIN_SIZE = 7000
HELD_OUT = 2000
GIVE_UP = 1e3

## SAVE DIRECTORY
DIR = CONFIG.DIR_TF_CKPTS

def training_model(model, data, SAVE_DIR):
    held_out, train_set = data[:HELD_OUT], data[HELD_OUT:]
    x = tf.placeholder(tf.float32, [None, DIM], name="subsample")
    loss = model.loss(x)
    ploss = model.ploss(x, BETA)
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(ploss)
    saver = tf.train.Saver(tf.trainable_variables(model.name))
    minLoss = 1e10
    noChange = 0
    with tf.Session() as sess:
        print(" Training ")
        sess.run([tf.global_variables_initializer(),
                tf.local_variables_initializer()])
        tf.get_default_graph().finalize()
        for i in range(N_EPOCHS):
            with util.NumpySeedContext(i):
                train_set = np.random.permutation(train_set)
            for j in range(int(TRAIN_SIZE/BATCH_SIZE)):
                subsample = train_set[j*BATCH_SIZE:(j+1) * BATCH_SIZE,:]
                _, = sess.run([opt], feed_dict={"subsample:0":subsample})
            ## Early stopping
            val, pval = sess.run([loss, ploss], feed_dict={"subsample:0":held_out})
            print(val, pval)
            minLoss = np.min([minLoss, val])
            if np.allclose(minLoss, val):
                print("{0} loss at epoch: {1}".format(minLoss, i))
                noChange = noChange + 1
                if not os.path.isdir(SAVE_DIR+"/{0}".format(i)):
                    os.mkdir(SAVE_DIR+"/{0}".format(i))
                saver.save(sess, SAVE_DIR+"/{0}/model".format(i),write_meta_graph=False)
            else:
                noChange = 0
            if noChange > GIVE_UP:
                break;
        print(" END and SAVE {0}".format(i))
        if not os.path.isdir(SAVE_DIR+"/end"):
            os.mkdir(SAVE_DIR+"/end")
        saver.save(sess, SAVE_DIR+"/end/model",write_meta_graph=False)

if __name__ == "__main__":
    n_layer = int(sys.argv[1])
    n_gauss = int(sys.argv[2])
    assert(n_gauss==5 or n_gauss==1)
    SAVE_DIR = ""
    # MKDIR
    SAVE_DIR =DIR+name+"/{0}_{1}".format(n_layer, n_gauss)
    if not os.path.isdir(SAVE_DIR):
        print("Creating Directory: {0}".format(DIR))
        os.mkdir(SAVE_DIR)
    loc,test_f = load_crime_dataset(c_type,TRAIN_SIZE)
    name = "{0}_{1}".format(n_layer, n_gauss)
    if n_gauss==1:
        model = OneGaussMAF(n_layer, name=name)
    else:
        model = FivGaussMAF(n_layer, name=name)
    loc = loc.astype(np.float32)
    training_model(model, loc, SAVE_DIR)
