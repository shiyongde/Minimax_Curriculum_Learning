from __future__ import print_function
from __future__ import division

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
import lasagne
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
#from scipy.spatial.distance import cdist
from scipy.sparse import issparse
#from sklearn.manifold import TSNE

from facloc_graph import facloc_graph
#from satcoverage import satcoverage
from concavefeature import concavefeature
#from setcover import setcover
from greedy import greedy
from randomGreedy import randomGreedy

def dataGroup(X, y, tsne_dim, num_cluster, dataset_name, savefile = True):

    print('clustering...')
    n = X.shape[0]
    y = numpy.asarray(y)
    class_label = numpy.unique(y)
    cluster_label = numpy.zeros(n)
    center_nn = numpy.array([])
    cluster_centers = ()
    startID = 0
    for i in class_label:
        iIndex = numpy.where(y==i)[0].astype(int)
        icluster_num = int(round(num_cluster*(len(iIndex)/n)))
        if tsne_dim <= 0 and not issparse(X):
            X_tsne = X[iIndex, :]
        elif issparse(X):
            print('TruncatedSVD of data size', len(iIndex))
            svd = TruncatedSVD(n_components=tsne_dim, algorithm='randomized', n_iter=10, random_state=42)
            X_tsne = svd.fit_transform(X[iIndex])
            print('finish TruncatedSVD.')
        else:
            print('PCA of data size', len(iIndex))
            pca = PCA(n_components = tsne_dim)
            X_tsne = pca.fit_transform(X[iIndex, :])
            print('finish PCA.')
        print('k-means', X_tsne.shape[0], 'points to', icluster_num, 'clusters')
        kmeans = KMeans(n_clusters = icluster_num).fit(X_tsne.astype('float64'))
        cluster_label[iIndex] = kmeans.labels_ + startID
        startID += icluster_num
        for j in range(icluster_num):
            jIndex = numpy.where(kmeans.labels_==j)[0]
            ijIndex = iIndex[jIndex]
            centerj = X[ijIndex].mean(axis=0)
            cluster_centers = cluster_centers + (centerj,)
            #center_nn = numpy.append(center_nn, ijIndex[numpy.argmin(cdist([centerj], X[ijIndex, :]))])
            center_nn = numpy.append(center_nn, ijIndex[numpy.argmin(euclidean_distances([kmeans.cluster_centers_[j]], X_tsne[jIndex]))])

    cluster_centers = numpy.vstack(cluster_centers)

    if savefile:
        numpy.savetxt(dataset_name + '_kmeans_labels.txt', cluster_label)
        numpy.savetxt(dataset_name + '_kmeans_centers.txt', cluster_centers)
        numpy.savetxt(dataset_name + '_center_nn.txt', center_nn)
        cluster_label, cluster_centers, center_nn = [],[],[]
    else:
        return cluster_label, cluster_centers, center_nn

def dataGroup0(X, tsne_dim, num_cluster, dataset_name, savefile = True):

    print('clustering...')
    n = X.shape[0]
    center_nn = numpy.array([])
    cluster_centers = ()
    if tsne_dim <= 0 and not issparse(X) and n <= 10000:
        X_tsne = X
    elif issparse(X) or n > 10000:
        if tsne_dim == 0:
            tsne_dim = 48
        print('TruncatedSVD of data size', (n, X.shape[1]))
        svd = TruncatedSVD(n_components=tsne_dim, algorithm='randomized', n_iter=10, random_state=42)
        X_tsne = svd.fit_transform(X)
        print('finish TruncatedSVD.')
    else:
        print('PCA of data size', n)
        pca = PCA(n_components = tsne_dim)
        X_tsne = pca.fit_transform(X)
        print('finish PCA.')
    print('k-means to', num_cluster, 'clusters')
    kmeans = KMeans(n_clusters = num_cluster, max_iter = 50).fit(X_tsne.astype('float64'))
    cluster_label = kmeans.labels_
    for j in range(num_cluster):
        jIndex = numpy.where(cluster_label==j)[0]
        centerj = numpy.mean(X[jIndex, :], axis = 0)
        cluster_centers = cluster_centers + (centerj,)
        center_nn = numpy.append(center_nn, jIndex[numpy.argmin(euclidean_distances([kmeans.cluster_centers_[j]], X_tsne[jIndex]))])

    cluster_centers = numpy.vstack(cluster_centers)

    if savefile:
        numpy.savetxt(dataset_name + '_kmeans_labels.txt', cluster_label)
        numpy.savetxt(dataset_name + '_kmeans_centers.txt', cluster_centers)
        numpy.savetxt(dataset_name + '_center_nn.txt', center_nn)
        cluster_label, cluster_centers, center_nn = [],[],[]
    else:
        return cluster_label, cluster_centers, center_nn

def initSubmodularFunc(X, k, func = 'concavefeature', func_parameter = 0.56):

    g = greedy(X, func = func, func_parameter = func_parameter, save_memory = [False, 8])
    V = g.V
    f = g.f
    nn, V_obj = f.evaluateV()
    minGain = V_obj - numpy.asarray([f.evaluate_decremental(nn, x, V)[1] for x in V])
    sinGain = numpy.asarray([f.evaluate([x])[1] for x in V])
    topkObj = sum(numpy.partition(sinGain, -k)[-k:])
    kcluster, greedyObj, f_Vsize = g(k)
    optObj = min([greedyObj/(1 - 1/numpy.e), topkObj])
    return minGain, sinGain, optObj

def pruneGroundSet(minGain, sinGain, loss_vec, k):

    # prune
    minGain = minGain + loss_vec
    sinGain = sinGain + loss_vec
    left_index = numpy.where(sinGain >= min(numpy.partition(minGain, -k)[-k:]))[0]
    #if len(left_index) <= k:
        #print('ERROR!', numpy.where(minGain - sinGain > 0)[0])
    # permutation
    minGain = minGain[left_index]
    sinGain = sinGain[left_index]
    left_index = left_index[(minGain/sinGain).argsort()]
    return left_index

def modularLowerBound(X, k, func, func_parameter, offset, optObj, approx = 0.5, iters = 5, mntone = True):

    #print('computing subdifferentials...')
    runGreedy = True
    if mntone:
        g = greedy(X, func = func, func_parameter = func_parameter, save_memory = [False, 8], offset = offset)
    else:
        g = randomGreedy(X, func = func, func_parameter = func_parameter, save_memory = [False, 8], offset = offset)
    perm = numpy.arange(X.shape[0])
    topkIndex_old = []

    #run subdifferetial for iters steps
    for it in range(iters):

        score = numpy.zeros(X.shape[0])
        nn, score[0] = g.f.evaluate([perm[0]])
        for i in range(1, X.shape[0]):
            nn, score[i] = g.f.evaluate_incremental(nn, perm[i])
        score[1:] = score[1:] - score[:-1]
        score += offset[perm]
        topkIndex = numpy.argpartition(score, -k)[-k:]
        mlb = sum(score[topkIndex])
        topkIndex = perm[topkIndex]
        perm = numpy.append(topkIndex, numpy.setxor1d(numpy.arange(X.shape[0]), topkIndex))

        # use random permutation if cannot improve
        if len(numpy.setxor1d(topkIndex, topkIndex_old)) == 0:
            #break
            perm = numpy.append(topkIndex, numpy.random.permutation(numpy.setxor1d(numpy.arange(X.shape[0]), topkIndex)))
        #test if approximation factor is large enough
        elif g.f.evaluate(topkIndex)[1] + sum(offset[topkIndex]) >= approx * optObj:
            runGreedy = False
            break
        else:
            topkIndex_old = topkIndex

    if runGreedy:
        #print('running greedy to update A')
        topkIndex, greedyObj, f_Vsize = g(k)
        #print('Approx factor =', greedyObj/optObj)

    return topkIndex

def load_mnist():

    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    #import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 28**2)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / numpy.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = numpy.frombuffer(f.read(), numpy.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    train_set_x = load_mnist_images('train-images-idx3-ubyte.gz')
    train_set_y = load_mnist_labels('train-labels-idx1-ubyte.gz')
    test_set_x = load_mnist_images('t10k-images-idx3-ubyte.gz')
    test_set_y = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    train_set_x, valid_set_x = train_set_x[:-10000], train_set_x[-10000:]
    train_set_y, valid_set_y = train_set_y[:-10000], train_set_y[-10000:]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)

def load_cifar10_1():

    def load_CIFAR_batch(filename):
      """ load single batch of cifar """
      with open(filename, 'r') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = numpy.array(Y)
        return X, Y

    def load_CIFAR10(ROOT):
      """ load all of cifar """
      xs = []
      ys = []
      for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
      Xtr = numpy.concatenate(xs)
      Ytr = numpy.concatenate(ys)
      del X, Y
      Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
      return Xtr, Ytr, Xte, Yte

    def get_CIFAR10_data(num_training=49000, num_val=1000, num_test=10000, show_sample=True):
        """
        Load the CIFAR-10 dataset, and divide the sample into training set, validation set and test set
        """

        cifar10_dir = '/home/tianyizhou/Downloads/cifar-10-batches-py'
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
            
        # subsample the data for validation set
        mask = xrange(num_training, num_training + num_val)
        X_val = X_train[mask]
        y_val = y_train[mask]
        mask = xrange(num_training)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = xrange(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def preprocessing_CIFAR10_data(X_train, X_val, X_test):
        
        # normalize to zero mean and unity variance
        offset = numpy.mean(X_train, 0)
        scale = numpy.std(X_train, 0).clip(min=1)
        X_train = (X_train - offset) / scale
        X_valid = (X_valid - offset) / scale
        X_test = (X_test - offset) / scale

        return X_train, X_val, X_test

    X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw = get_CIFAR10_data()
    X_train, X_val, X_test= preprocessing_CIFAR10_data(X_train, X_val, X_test)

    return (X_train, y_train_raw), (X_val, y_val_raw), (X_test, y_test_raw)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar10_2(tsne_dim = 300, num_cluster = 2000, dataset_name = 'cifar10'):
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('/home/tianyizhou/Downloads/cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('/home/tianyizhou/Downloads/cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = numpy.concatenate(xs)/numpy.float32(255)
    y = numpy.concatenate(ys)

    #X_fea = numpy.loadtxt('/home/tianyizhou/Downloads/kmeans-learning-torch-master/cifar10_500fea.txt')
    #X_fea = X_fea[0:50000, :]
    #dataGroup(x[0:50000], y[0:50000], tsne_dim, num_cluster, dataset_name, savefile=True)
    labels_ = numpy.loadtxt(dataset_name + '_kmeans_labels.txt').astype(int)
    cluster_centers_ = numpy.loadtxt(dataset_name + '_kmeans_centers.txt')
    center_nn = numpy.loadtxt(dataset_name + '_center_nn.txt').astype(int)
    center_nn_flip = center_nn + 50000
    center_nn = numpy.concatenate((center_nn,center_nn_flip),axis=0)

    x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = numpy.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = numpy.concatenate((X_train,X_train_flip),axis=0)
    Y_train = numpy.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),
        #X_fea = lasagne.utils.floatX(X_fea),
        kmeans_label = labels_.astype('int32'),
        kmeans_center = lasagne.utils.floatX(cluster_centers_),
        kmeans_center_nn = center_nn.astype('int32'),)

def load_cifar100(tsne_dim = 300, num_cluster = 2000, dataset_name = 'cifar100'):
   
    d = unpickle('/home/tianyizhou/Downloads/cifar-100-python/train')
    x = d['data']
    y0 = d['coarse_labels']
    y1 = d['fine_labels']

    d = unpickle('/home/tianyizhou/Downloads/cifar-100-python/test')

    x = numpy.concatenate((x, d['data']))/numpy.float32(255)
    y0 = numpy.concatenate((y0, d['coarse_labels']))
    y1 = numpy.concatenate((y1, d['fine_labels']))

    x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = numpy.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y1[0:50000]
    X_test = x[50000:,:,:,:]
    Y_test = y1[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

def load_cifar10_3(tsne_dim = 0, num_cluster = 2000, dataset_name = 'cifar10'):
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('/home/tianyizhou/Downloads/cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('/home/tianyizhou/Downloads/cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = numpy.concatenate(xs)/numpy.float32(255)
    y = numpy.concatenate(ys)

    #X_fea = numpy.loadtxt('/home/tianyizhou/Downloads/kmeans-learning-torch-master/cifar10_500fea.txt')
    #X_fea = X_fea[0:50000, :]
    #dataGroup(x[0:50000], y[0:50000], tsne_dim, num_cluster, dataset_name)
    #labels_ = numpy.loadtxt(dataset_name + '_kmeans_labels.txt').astype(int)
    #cluster_centers_ = numpy.loadtxt(dataset_name + '_kmeans_centers.txt')
    #center_nn = numpy.loadtxt(dataset_name + '_center_nn.txt').astype(int)
    #center_nn_flip = center_nn + 50000
    #center_nn = numpy.concatenate((center_nn,center_nn_flip),axis=0)

    x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = numpy.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = numpy.concatenate((X_train,X_train_flip),axis=0)
    Y_train = numpy.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)
        ##X_fea = lasagne.utils.floatX(X_fea),
        #kmeans_label = labels_.astype('int32'),
        #kmeans_center = lasagne.utils.floatX(cluster_centers_),
        #kmeans_center_nn = center_nn.astype('int32'),)



def load_data_old(dataset_name = 'cifar10'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    load_cifar10_2()

    tsne_dim = 200
    num_cluster = 1000
    #train_set, valid_set, test_set = load_mnist1()
    #train_set, valid_set, test_set = load_cifar10()
    #dataGroup(train_set[0], train_set[1], tsne_dim, num_cluster)    
    #labels_ = numpy.loadtxt('cifar10_kmeans_labels.txt').astype(int)
    #cluster_centers_ = numpy.loadtxt('cifar10_kmeans_centers.txt')
    #print(numpy.unique(labels_), cluster_centers_.shape)

    train_set, valid_set, test_set = load_mnist2()
    #dataGroup(train_set[0], train_set[1], tsne_dim, num_cluster, dataset_name)
    labels_ = numpy.loadtxt(dataset_name + '_kmeans_labels.txt').astype(int)
    cluster_centers_ = numpy.loadtxt(dataset_name + '_kmeans_centers.txt')
    center_nn = numpy.loadtxt(dataset_name + '_center_nn.txt').astype(int)

    #print(test_set)

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y), (labels_, cluster_centers_, center_nn)]
    return rval

    # ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    len_inputs = len(inputs)
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, numpy.ceil(len_inputs/batchsize).astype('int32')*batchsize, batchsize):
        if start_idx + batchsize > len_inputs:
            end_index = len_inputs
            batchsize = len_inputs - start_idx
        else:
            end_index = start_idx + batchsize
        if shuffle:
            excerpt = indices[start_idx:end_index]
        else:
            excerpt = slice(start_idx, end_index)
        if augment:
            # as in paper : 
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = numpy.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = numpy.zeros(inputs[excerpt].shape, dtype=numpy.float32)
            crops = numpy.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]

def iterate_minibatches0(inputs, batchsize):

    len_inputs = len(inputs)
    for start_idx in range(0, numpy.ceil(len_inputs/batchsize).astype('int32')*batchsize, batchsize):
        if start_idx + batchsize > len_inputs:
            end_index = len_inputs
            batchsize = len_inputs - start_idx
        else:
            end_index = start_idx + batchsize
        excerpt = slice(start_idx, end_index)

        yield inputs[excerpt]
