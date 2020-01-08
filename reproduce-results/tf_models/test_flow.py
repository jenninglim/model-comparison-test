import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from flow import FlowModel
from kgof import glo
import os

tfb = tfp.bijectors
tfd = tfp.distributions
os.getcwd()
# Data distribution
dd = np.load(glo.data_file(os.getcwd()+'/../../data/chicago_crime_loc_with_type2016.npz'))['data']
year = 2016
c_type = 'ROBBERY'
size = 1100
def filter_crimetype(data, type = None):
    if type is None:
        data = data
    else:
        data = data[data[:,0] == type]
    if len(data) == 1:
        print("No Crime Type found")
    else:
        loc = data[:,1:].astype(float)
        loc = np.nan_to_num(loc)
        loc = loc[loc[:,0] != 0]
        #Set City bound
        loc = loc[loc[:,0] >-89]
        loc = loc[loc[:,1] > 40]
        return loc
loc = filter_crimetype(dd, c_type)

loc.shape
# Model Parameters
dims = 2
n_layers = 10
model = FlowModel(dims, n_layers)

# Data Samples
opt = tf.train.AdamOptimizer(1e-4).minimize(model.loss(loc))

tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
for i in range(1000):
    if i % 100 == 0:
        print(model.loss(loc).eval())
    opt.run()
#tf.trainable_variables()
fake_samples = model.sample(100).eval()
plt.scatter(loc[:,0],loc[:,1])
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
X,Y = np.meshgrid(x,y)
mat = np.dstack([X,Y])
plt.contour(X,Y, np.exp(model.log_prob(mat).eval()))
saver.save(tf.get_default_session(), "./model.ckpt")
