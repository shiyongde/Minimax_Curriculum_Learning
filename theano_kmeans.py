"""
A simple example of convolutional K-means based on Theano
"""

import os;
import gzip;
import cPickle as pickle;

import numpy as np;
import theano;
import theano.tensor as T;
#from theano.tensor.nnet.signal import pool as downsample;
from theano.tensor.nnet import conv;

import matplotlib;
#from tensorflow.models.image.mnist.convolutional import NUM_CHANNELS
matplotlib.use('tkagg');
import matplotlib.pyplot as plt;


import util;

### Utility functions

def load_mnist(dataset):
    """Load MNIST dataset
    
    Parameters
    ----------
    dataset : string
        address of MNIST dataset
    
    Returns
    -------
    rval : list
        training, valid and testing dataset files
    """
    
    # Load the dataset
    f = gzip.open(dataset, 'rb');
    train_set, _, _ = pickle.load(f);
    f.close();
    
    X=train_set[0].T;
    X=util.normalize_data(X);
    X=util.ZCA_whitening(X);
    
    return X.T;

class ConvKmeans(object):
  """ a simple implementation of Conv K-means Class, not a general one"""
  def __init__(self,
               filter_size,
               num_filters,
               num_channels,
               fm_size=None,
               batch_size=None,
               border_mode="valid"):
    
    self.filter_size=filter_size;
    self.num_filters=num_filters;
    self.num_channels=num_channels;
    self.fm_size=fm_size;
    self.batch_size=batch_size;
    self.border_mode=border_mode;
    
    self.initialize()
    
  def initialize(self):
    
    D=np.random.normal(size=(self.filter_size[0]*self.filter_size[1],
                             self.num_filters));
    D=D/np.sqrt(np.sum(D**2, axis=0));
    D=D.reshape(self.filter_size[0],
                self.filter_size[1],
                1,
                self.num_filters).transpose(3,2,0,1);
                
    self.filters=theano.shared(np.asarray(D,
                                    dtype=theano.config.floatX),
                         borrow=True);

  def apply(self, X):
    Y=conv.conv2d(input=X,
                  filters=self.filters,
                  image_shape=(self.batch_size, self.num_channels)+(self.fm_size),
                  filter_shape=(self.num_filters, self.num_channels)+(self.filter_size),
                  border_mode=self.border_mode);
    
    return Y;

def kmeans_train(datasets = "../data/mnist.pkl.gz", batch_size=1000, num_filters=100)

### Data preparation

  data=load_mnist(datasets);
  isize = sqrt(data.shape[1])
  shared_data=theano.shared(np.asarray(data,
                                       dtype=theano.config.floatX),
                            borrow=True);

  n_train_batches=floor(data.shape[0]/batch_size);

  print "[MESSAGE] The data is loaded"
  print "[MESSAGE] Building model"

  idx=T.lscalar();
  X=T.matrix('x');

  images=X.reshape((batch_size, 1, 28, 28));

layer=ConvKmeans(filter_size=(28,28),
                 num_filters=num_filters,
                 num_channels=1,
                 fm_size=(28, 28),
                 batch_size=batch_size);
                 
out=layer.apply(images);
print(out)
out=out*(out>=T.max(out, axis=1, keepdims=True));
filters_new=conv.conv2d(input=images.dimshuffle(1,0,2,3),
                        filters=out.dimshuffle(1,0,2,3),
                        image_shape=(1,batch_size,28,28),
                        filter_shape=(num_filters, batch_size, 1, 1),
                        border_mode="valid");
filters_new=filters_new.dimshuffle(1,0,2,3);
                       
#updates=[(layer.filters, (layer.filters+filters_new)/(T.sum((layer.filters+filters_new)**2, axis=0, keepdims=True)))];

f=theano.function(inputs=[idx],
                  outputs=filters_new,
                  givens={X: shared_data[idx * batch_size: (idx + 1) * batch_size]});

for i in xrange(10):
  for k in xrange(50):
    y=f(k);
    print(y.shape)
    filters=layer.filters.get_value(borrow=True);
    filters=filters+y;
    filters=filters/np.sqrt(np.array([np.sum(filters[j]**2, keepdims=True) for j in xrange(filters.shape[0])]));
    layer.filters.set_value(filters);
    print "[MESSAGE] Internal Iteration %i is done" % (k);
  
  print "[MESSAGE] Iteration %i is done" % (i);
                   

filters=layer.filters.get_value(borrow=True);
print filters.shape;

plt.figure(1);
for i in xrange(num_filters):
  plt.subplot(10,10,i+1);  
  plt.imshow(filters[i,0,:,:], cmap = plt.get_cmap('gray'), interpolation='nearest');
  plt.axis('off')
     
plt.show();