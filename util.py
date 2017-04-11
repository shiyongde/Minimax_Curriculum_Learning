"""
Some utility functions
Author: Yuhuang Hu
Email: duguyue100@gmail.com
"""

import os;
import gzip;
import cPickle as pickle;

import numpy as np;
import numpy.linalg as LA;


### DATA LOADING FUCNTIONS ###

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
    train_set, valid_set, test_set = pickle.load(f);
    f.close();
  
    #mean_image=get_mean_image(train_set[0]);

    return [train_set, valid_set, test_set];
  
def load_CIFAR_batch(filename):
    """
    load single batch of cifar-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param filename: string of file name in cifar
    @return: X, Y: data and labels of images in the cifar batch
    """
    
    with open(filename, 'r') as f:
        datadict=pickle.load(f);
        
        X=datadict['data'];
        Y=datadict['labels'];
        
        X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float");
        Y=np.array(Y);
        
        return X, Y;
      
def load_CIFAR10(ROOT):
    """
    load entire CIFAR-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param ROOT: string of data folder
    @return: Xtr, Ytr: training data and labels
    @return: Xte, Yte: testing data and labels
    """
    
    xs=[];
    ys=[];
    
    for b in range(1,6):
        f=os.path.join(ROOT, "data_batch_%d" % (b, ));
        X, Y=load_CIFAR_batch(f);
        xs.append(X);
        ys.append(Y);
        
    Xtr=np.concatenate(xs);
    Ytr=np.concatenate(ys);
    
    del X, Y;
    
    Xte, Yte=load_CIFAR_batch(os.path.join(ROOT, "test_batch"));
    
    return Xtr, Ytr, Xte, Yte;
  
### UTILITIES FUNCTIONS ###  

def sample_patches_mnist(data, N, d):
  """
  Sample image patches from mnist
  """
  X=np.array([]);
  for i in xrange(N):
    temp=data[0][i].reshape(28,28);
    temp=temp[5:5+d, 5:5+d].reshape(-1);
    if X.size is 0:
      X=temp;
    else:
      X=np.vstack((X, temp));
  
  return X.T;

def sample_patches_cifar10(data, N, d):
  """
  Sample image patches from CIFAR-10
  """
  
  X=np.array([]);
  
  for i in xrange(N):
    temp=data[i, :, :];
    temp=temp[5:5+d, 5:5+d].reshape(-1);
    if X.size is 0:
      X=temp;
    else:
      X=np.vstack((X, temp));
  
  return X.T;

def normalize_data(X):
  """
  Normalize data's mean and variance,
  
  X in (data_size x number_of_samples) manner
  """
  
  X_mean=np.mean(X, axis=0);
  X_var=np.var(X, axis=0)+10;
  X=(X-X_mean)/np.sqrt(X_var);
  
  return X;

def ZCA_whitening(X):
  """
  Perform ZCA_whitening
  
  X in (data_size x number_of_samples) manner
  """
  
  U, S, _ = LA.svd(X.dot(X.T)/X.shape[1]);
  
  return U.dot(np.diag(1/np.sqrt(S+0.01))).dot(U.T).dot(X);

def kmeans(X, D, num_iter):
  """
  Perform kmeans algorithm
  
  Parameters
  ----------
  X : 2-D array with size of (data_size x num_of_samples);
      image patches
      
  D : 2-D array with size of (data_size x num_of_centroids);
      dictioanry
      
  num_iter : integer
      number of iteration
  """
  
  for i in xrange(num_iter):
    S=D.T.dot(X);
    S=S*(S>=np.max(S, axis=0));
    D=X.dot(S.T)+D;
    D=D/np.sqrt(np.sum(D**2, axis=0));
    print "[MESSAGE] Iteration %i is done" % (i);
    
  return D;