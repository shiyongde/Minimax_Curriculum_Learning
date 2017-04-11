"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

from __future__ import print_function
from __future__ import division

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
#from sklearn.manifold import TSNE

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from mlp import HiddenLayer

import matplotlib.pyplot as plt

from facloc_graph import facloc_graph
#from satcoverage import satcoverage
from concavefeature import concavefeature
#from setcover import setcover
from greedy import greedy

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W0 = [], b0 = []):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if len(W0) == 0 or len(b0) == 0:
            W0 = numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            )
            b0 = numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )

        self.W = theano.shared(
            value=W0,
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=b0,
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood_vec(self, y):
        return -T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

def dataGroup(X, y, tsne_dim, num_cluster, dataset_name):

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
        print('PCA of data size', len(iIndex))
        pca = PCA(n_components = tsne_dim)
        X_tsne = pca.fit_transform(X[iIndex, :])
        #model = TSNE(n_components=tsne_dim, init = 'pca', random_state=0)
        #X_tsne = model.fit_transform(X[iIndex, :])
        print('finish PCA.')
        print('k-means to', icluster_num, 'clusters')
        kmeans = KMeans(n_clusters = icluster_num).fit(X_tsne)
        cluster_label[iIndex] = kmeans.labels_ + startID
        startID += icluster_num
        for j in range(icluster_num):
            ijIndex = iIndex[kmeans.labels_==j]
            centerj = numpy.mean(X[ijIndex, :], axis = 0)
            center_nn = numpy.append(center_nn, ijIndex[numpy.argmin(cdist([centerj], X[ijIndex, :]))])
            cluster_centers = cluster_centers + (centerj,)

    cluster_centers = numpy.vstack(cluster_centers)

    numpy.savetxt(dataset_name + '_kmeans_labels.txt', cluster_label)
    numpy.savetxt(dataset_name + '_kmeans_centers.txt', cluster_centers)
    numpy.savetxt(dataset_name + '_center_nn.txt', center_nn)

def load_mnist1():

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    return train_set, valid_set, test_set

def load_mnist2():

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
    import gzip

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

def load_cifar10():

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

        cifar10_dir = '/Users/tianyi/Downloads/cifar-10-batches-py'
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
        
        # Preprocessing: reshape the image data into rows
        X_train = numpy.reshape(X_train, (X_train.shape[0], -1)) # [49000, 3072]
        X_val = numpy.reshape(X_val, (X_val.shape[0], -1)) # [1000, 3072]
        X_test = numpy.reshape(X_test, (X_test.shape[0], -1)) # [10000, 3072]
        
        # Normalize the data: subtract the mean image
        mean_image = numpy.mean(X_train, axis = 0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        
        return X_train, X_val, X_test

    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
    X_train, X_val, X_test= preprocessing_CIFAR10_data(X_train_raw, X_val_raw, X_test_raw)

    return (X_train, y_train_raw), (X_val, y_val_raw), (X_test, y_test_raw)

def load_cifar10_fea():

    X = numpy.loadtxt('/home/tianyizhou/Downloads/kmeans-learning-torch-master/cifar10_500fea.txt')
    y = numpy.loadtxt('cifar10_label.txt')
    X_train = X[0:50000]
    y_train = y[0:50000]
    X_val = X[49000:50000]
    y_val = y[49000:50000]
    X_test = X[50000:]
    y_test = y[50000:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    tsne_dim = 200
    num_cluster = 1000
    dataset_name = 'cifar10'
    #train_set, valid_set, test_set = load_mnist1()
    #train_set, valid_set, test_set = load_cifar10()
    #dataGroup(train_set[0], train_set[1], tsne_dim, num_cluster)    
    #labels_ = numpy.loadtxt('cifar10_kmeans_labels.txt').astype(int)
    #cluster_centers_ = numpy.loadtxt('cifar10_kmeans_centers.txt')
    #print(numpy.unique(labels_), cluster_centers_.shape)

    train_set, valid_set, test_set = load_cifar10_fea()
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

def initSubmodularFunc(X, k, func = 'concavefeature', func_parameter = 0.5):

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

def modularLowerBound(X, k, func, func_parameter, offset, optObj, approx = 0.5, iters = 3):

    #print('computing subdifferentials...')
    runGreedy = True
    g = greedy(X, func = func, func_parameter = func_parameter, save_memory = [False, 8], offset = offset)
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
        print('running greedy to update A')
        topkIndex, greedyObj, f_Vsize = g(k)
        print('Approx factor =', greedyObj/optObj)

    return topkIndex

def sgd_optimization_mnist(learning_rate=6e-2, loss_weight = 2.6e+3, curriculum_rate=0.03, n_curriculum_epochs=300, epoch_iters = 20, converge = 1e-4,
                           minibatch_size = 100, 
                           batch_size=8, k = 8, func = 'concavefeature', func_parameter = 0.5, deep = False):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    print('loading data...')
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    labels_, cluster_centers_, center_nn = datasets[3]
    num_cluster = cluster_centers_.shape[0]
    #n_channels = 3
    #isize = int(numpy.sqrt(train_set_x.get_value(borrow=True).shape[1]/n_channels))
    n_fea = train_set_x.get_value(borrow=True).shape[1]

    # compute number of minibatches for training, validation and testing
    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_train // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('building the model...')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    cindex = T.lvector()  # index to a [mini]batch


    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    if deep is False:

        # construct the logistic regression class
        # Each MNIST image has size 28*28
        classifier = LogisticRegression(input=x, n_in=n_fea, n_out=10)

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = classifier.negative_log_likelihood(y)
        cost_vec = classifier.negative_log_likelihood_vec(y)

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)

        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                   (classifier.b, classifier.b - learning_rate * g_b)]
    else:

        nfea = 1000
        nkerns=[32, 32, 64]
        rng = numpy.random.RandomState(23455)

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((-1, n_channels, isize, isize))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(None, n_channels, isize, isize),
            filter_shape=(nkerns[0], n_channels, 3, 3),
            poolsize=(2, 2)
        )

        isize1 = int((isize - 3 + 1)/2)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(None, nkerns[0], isize1, isize1),
            filter_shape=(nkerns[1], nkerns[0], 4, 4),
            poolsize=(2, 2)
        )

        isize2 = int((isize1 - 4 + 1)/2)

        layer2 = LeNetConvPoolLayer(
            rng,
            input=layer1.output,
            image_shape=(None, nkerns[1], isize2, isize2),
            filter_shape=(nkerns[2], nkerns[1], 5, 5),
            poolsize=(2, 2)
        )

        isize3 = int((isize2 - 5 + 1)/2)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer3_input = layer2.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer3 = HiddenLayer(
            rng,
            input=layer3_input,
            n_in=nkerns[2] * isize3 * isize3,
            n_out=nfea,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        classifier = LogisticRegression(input=layer3.output, n_in=nfea, n_out=10)

        # the cost we minimize during training is the NLL of the model
        cost = classifier.negative_log_likelihood(y)
        cost_vec = classifier.negative_log_likelihood_vec(y)

        # create a list of all model parameters to be fit by gradient descent
        params = classifier.params + layer3.params + layer2.params + layer1.params + layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[cindex],
        outputs=classifier.errors(y),
        updates=updates,
        givens={
            x: train_set_x[cindex],
            y: train_set_y[cindex]
        }
    )

    loss_model = theano.function(
        inputs=[cindex],
        outputs=cost_vec,
        givens={
            x: train_set_x[cindex],
            y: train_set_y[cindex]
        }
    )
    error_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('training the model...')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    #validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    #initialize
    minGain, sinGain, optSubmodular = initSubmodularFunc(cluster_centers_, k)
    real_iter = 0
    validation_frequency = 20
    old_epoch_all_loss = float('inf')
    loss_weight0 = loss_weight
    passed_index = numpy.array([])
    passed_index_epoch = numpy.array([]) 
    passes = 0
    output_seq = ()
    memory_size = 100
    forget_weight = 80
    forget = numpy.ones(num_cluster) * memory_size
    for curriculum_epoch in range(n_curriculum_epochs):

        print('Epoch', curriculum_epoch)
        old_all_loss = 0
        for iters in range(epoch_iters):

            if len(passed_index) <= n_train*0.35:
                # compute loss
                loss_vec = loss_model(center_nn) * loss_weight / len(center_nn)
                all_loss = sum(loss_vec)
                #loss_vec_center = numpy.asarray([sum(loss_vec[labels_ == i]) for i in range(num_cluster)])
                loss_vec_center = loss_vec + forget
                topkLoss = sum(numpy.partition(loss_vec_center, -k)[-k:])
                print(optSubmodular, topkLoss)
                optObj = optSubmodular + topkLoss

                # update A (topkIndex)
                left_index = pruneGroundSet(minGain, sinGain, loss_vec_center, k)
                topkIndex = modularLowerBound(cluster_centers_[left_index,:], k, func, func_parameter, loss_vec_center[left_index], optObj)
                topkIndex = left_index[topkIndex]
                forget[topkIndex] -= 2
                forget[forget <= 0] = 100

                # update classifier (train_model)           
                train_index = numpy.array([])
                for i in range(len(topkIndex)):
                    train_index = numpy.append(train_index, numpy.where(labels_ == topkIndex[i])[0])
                train_index = numpy.random.permutation(train_index.astype(int))
                print('number of training samples =', len(train_index))
                passes += len(train_index)
                passed_index = numpy.unique(numpy.append(passed_index, train_index))
                passed_index_epoch = numpy.unique(numpy.append(passed_index_epoch, train_index))

            else:

                train_index = numpy.random.permutation(numpy.setxor1d(numpy.arange(n_train), passed_index_epoch).astype(int))
                #train_index = numpy.random.permutation(numpy.arange(n_train).astype(int))
                passes += len(train_index)
                passed_index_epoch = numpy.array([])
                #passed_index = numpy.arange(n_train)
                
            # training by mini-batch sgd
            start_index = 0
            train_loss = numpy.array([])
            while start_index < len(train_index):
                end_index = min([start_index + minibatch_size, len(train_index)])
                batch_index = train_index[start_index : end_index]
                start_index += end_index
                train_loss = numpy.append(train_loss, train_model(batch_index))
            this_train_loss = numpy.mean(train_loss)

            # stop the current epoch if converge
            diff_loss = old_all_loss - all_loss
            if  diff_loss >= 0 and diff_loss <= all_loss * converge:
                break
            # show validation and test error peoriodically
            else:
                old_all_loss = all_loss
                if (iters + real_iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    test_losses = [test_model(i)
                                    for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    train_score = [error_model(i) 
                                    for i in range(n_train_batches)]
                    this_train_score = numpy.mean(train_score)

                    print(
                        'minibatch %i, %i trainings, %i passes, trainErr %f %%, validErr %f %%, testErr %f %%' %
                        (
                            iters + real_iter + 1,
                            len(passed_index),
                            passes,
                            this_train_score * 100.,
                            this_validation_loss * 100.,
                            test_score * 100.
                        )
                    )

                    output_seq = output_seq + (numpy.array([len(passed_index),passes,this_train_score * 100.,this_validation_loss * 100.,test_score * 100.]),)

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, (iters + real_iter + 1) * patience_increase)

                        best_validation_loss = this_validation_loss

                        # save the best model
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)

        #print('Up to now %i training samples are used'%(len(passed_index)))
        # record total number of iterations
        real_iter += iters
        # adjust learning rate
        if all_loss > 1.001 * old_epoch_all_loss:
            print('no improvement: reduce learning rate!')
            learning_rate *= 0.9
        old_epoch_all_loss = all_loss
        # increase curriculum rate
        loss_weight *= curriculum_rate + 1

        if patience <= iters + real_iter + 1:
            break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    #print('The code run for %d epochs, with %f epochs/sec' % (
        #epoch, 1. * epoch / (end_time - start_time)))
    #print(('The code for file ' +
           #os.path.split(__file__)[1] +
           #' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
    output_seq = numpy.vstack(output_seq)
    return output_seq

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':

    output_seq = sgd_optimization_mnist()
    numpy.savetxt('cifar10_500fea_result.txt', output_seq)

    plt.figure(figsize = (20, 10))
    plt.subplot(1,2,1)
    plt.plot(output_seq[:, 1], output_seq[:, 2], 'yo-', label = 'training error')
    plt.plot(output_seq[:, 1], output_seq[:, 3], 'co-', label = 'validation error')
    plt.plot(output_seq[:, 1], output_seq[:, 4], 'mo-', label = 'test error')
    plt.grid()
    plt.legend(fontsize='large', loc = 1)
    plt.ylabel('Error rate (%)')   
    plt.xlabel('Number of passed training samples (including copies)')

    plt.subplot(1,2,2)
    plt.plot(output_seq[:, 0], output_seq[:, 2], 'yo-', label = 'training error')
    plt.plot(output_seq[:, 0], output_seq[:, 3], 'co-', label = 'validation error')
    plt.plot(output_seq[:, 0], output_seq[:, 4], 'mo-', label = 'test error')
    plt.grid()
    plt.legend(fontsize='large', loc = 1)
    plt.ylabel('Error rate (%)') 
    plt.xlabel('Size of set of passed training samples')
   
    plt.savefig('clearn_cifar10.eps', format = 'eps', bbox_inches='tight')
    plt.show()