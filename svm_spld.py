#!/usr/bin/env python

"""
Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)
Check the accompanying files for pretrained models. The 32-layer network (n=5), achieves a validation error of 7.42%, 
while the 56-layer network (n=9) achieves error of 6.75%, which is roughly equivalent to the examples in the paper.
"""

from __future__ import print_function
from __future__ import division

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import scipy.sparse as sp
import theano
import theano.tensor as T
from theano import sparse
#import lasagne
from cLearn_utils import *
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in current working directory

# ##################### Build the neural network model #######################

#from lasagne.layers import Conv2DLayer as ConvLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.layers import ElemwiseSumLayer
# from lasagne.layers import InputLayer
# from lasagne.layers import DenseLayer
# from lasagne.layers import GlobalPoolLayer
# from lasagne.layers import PadLayer
# from lasagne.layers import ExpressionLayer
# from lasagne.layers import NonlinearityLayer
# from lasagne.layers import FlattenLayer
# from lasagne.nonlinearities import softmax, rectify
# from lasagne.layers import batch_norm

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation = T.tanh, inputIsSparse=True):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        """
        self.input = input
        # if inputIsSparse:
        #     activation = sparse.tanh
        # else:
        #     activation = T.tanh

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        if inputIsSparse:
            lin_output = sparse.structured_dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W) + self.b

        # parameters of the model
        self.output = (lin_output if activation is None
                        else activation(lin_output)) 
        self.params = [self.W, self.b]

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, inputIsSparse=True):
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
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
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
        if inputIsSparse:
            self.p_y_given_x = T.nnet.softmax(sparse.structured_dot(input, self.W) + self.b)
        else:
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def cost_vec(self, y):
        return -T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]

    def cost(self, y):
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
            return T.mean(T.neq(self.y_pred, y)).astype(theano.config.floatX)
        else:
            raise NotImplementedError()

class OVASVMLayer(object):
    """SVM-like layer
    """
    def __init__(self, input, n_in, n_out, inputIsSparse = True):
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

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # parameters of the model
        self.params = [self.W, self.b]

        if inputIsSparse:
            self.output = sparse.structured_dot(input, self.W) + self.b
        else:
            self.output = T.dot(input, self.W) + self.b

        self.y_pred = T.argmax(self.output, axis=1)

    def hinge(self, u):
            return T.maximum(0, 1 - u)

    def cost_vec(self, y1):
        """ return the one-vs-all svm cost
        given ground-truth y in one-hot {-1, 1} form """
        y1_printed = theano.printing.Print('this is important')(T.max(y1))
        margin = y1 * self.output
        cost = self.hinge(margin).sum(axis=1)
        return cost

    def cost(self, y1):
        cost = self.cost_vec(y1).mean(axis=0)
        return cost

    def errors(self, y):
        """ compute zero-one loss
        note, y is in integer form, not one-hot
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def build_cnn(input_var, input_dim=100,
              fc_num_units=[], num_classes=20, rng=np.random.RandomState(23455), useSVM=True, 
              **junk):
    # input layer
    #network = lasagne.layers.InputLayer(shape=(None,) + (input_shape,),
    #                                    input_var=input_var)
    # fc-relu

        # construct a fully-connected sigmoidal layer
    for i in range(len(fc_num_units)):
        if i==0:
            network = HiddenLayer(
                rng,
                input=input_var,
                n_in=input_dim,
                n_out=fc_num_units[0],
                activation = T.tanh,
            )
            params=network.params
        else:
            network = HiddenLayer(
                rng,
                input=network.output,
                n_in=fc_num_units[i-1],
                n_out=fc_num_units[i],
                activation = T.tanh,
                inputIsSparse = False,
            )
            params = params + network.params            

    # output layer
    if len(fc_num_units) == 0:
        if useSVM:
            network = OVASVMLayer(input=input_var, n_in=input_dim, n_out=num_classes)
        else:
            network = LogisticRegression(input=input_var, n_in=input_dim, n_out=num_classes)
        params = network.params
    else:
        if useSVM:
            network = OVASVMLayer(input=network.output, n_in=fc_num_units[-1], n_out=num_classes, inputIsSparse=False)
        else:
            network = LogisticRegression(input=input_var, n_in=fc_num_units[-1], n_out=num_classes, inputIsSparse=False)
        params = params + network.params

    return network, params

def label_vec2mat(data_y):

    n_classes = len(np.unique(data_y))  # dangerous?
    y1 = -1 * np.ones((data_y.shape[0], n_classes))
    y1[np.arange(data_y.shape[0]), data_y] = 1

    return y1

def load20newsgroup(categories=None, filtered=False, use_hashing=False, tsne_dim=512, num_cluster=200, dataset_name='20newsgroup'):

    if filtered:
        remove = ('headers', 'footers', 'quotes')
    else:
        remove = ()

    print("Loading 20 newsgroups dataset for categories:")
    print(categories if categories else "all")

    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)

    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42,
                                   remove=remove)
    print('data loaded')

    # order of labels in `target_names` can be different from `categories`
    target_names = data_train.target_names


    def size_mb(docs):
        return sum(len(s.encode('utf-8')) for s in docs) / 1e6

    data_train_size_mb = size_mb(data_train.data)
    data_test_size_mb = size_mb(data_test.data)

    print("%d documents - %0.3fMB (training set)" % (
        len(data_train.data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(data_test.data), data_test_size_mb))

    # split a training set and a test set
    Y_train, Y_test = data_train.target, data_test.target

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time.time()
    if use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=opts.n_features)
        X_train = vectorizer.transform(data_train.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(data_train.data)
    duration = time.time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time.time()
    X_test = vectorizer.transform(data_test.data)
    duration = time.time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)

    #dataGroup(X_train, y_train, tsne_dim, num_cluster, dataset_name, savefile=True)
    #dataGroup0(X_train, tsne_dim, num_cluster, dataset_name, savefile=True)
    labels_ = np.loadtxt(dataset_name + '_kmeans_labels.txt').astype(int)
    cluster_centers_ = np.loadtxt(dataset_name + '_kmeans_centers.txt')
    center_nn = np.loadtxt(dataset_name + '_center_nn.txt').astype(int)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = sparse.shared(data_x.astype(theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),
                                 borrow=borrow)

        # one-hot encoded labels as {-1, 1}
        n_classes = len(np.unique(data_y))  # dangerous?
        y1 = -1 * np.ones((data_y.shape[0], n_classes))
        y1[np.arange(data_y.shape[0]), data_y] = 1
        shared_y1 = theano.shared(np.asarray(y1,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_y1,  'int32')

    #X_train, Y_train, Y1_train = shared_dataset((X_train, y_train))
    #X_test, Y_test, Y1_test = shared_dataset((X_test, y_test))

    return dict(
        X_train=X_train.astype(theano.config.floatX),
        Y_train=Y_train.astype('int32'),
        Y1_train=label_vec2mat(Y_train).astype('int32'),
        X_test=X_test.astype(theano.config.floatX),
        Y_test=Y_test.astype('int32'),
        Y1_test=label_vec2mat(Y_test).astype('int32'),
        kmeans_label = labels_.astype('int32'),
        kmeans_center = cluster_centers_.astype(theano.config.floatX),
        kmeans_center_nn = center_nn.astype('int32'),)

# ############################## Main program ################################

def main(num_epochs=50, model=None, 
        learning_rate=3.5, 
        epoch_iters = 25, batch_size = 64, 
        k = 4, useSVM = False, spld = [False, 4000, 4e-1, 3e-1, 1e-1]):

    # Load the dataset
    # print("Loading data...")
    data = load20newsgroup()
    X_train = data['X_train']
    #print(type(X_train.get_value(borrow=True)[range(2,5)]))
    Y_train = data['Y_train']
    Y1_train = data['Y1_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    Y1_test = data['Y1_test']
    labels_ = data['kmeans_label']
    labels_weight = np.array([len(np.where(labels_==i)[0]) for i in np.unique(labels_)])
    labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))

    cluster_centers_ = data['kmeans_center']
    center_nn = data['kmeans_center_nn']
    num_cluster = cluster_centers_.shape[0]

    # compute number of minibatches for training, validation and testing
    n_train = X_train.shape[0]
    n_train_batches = n_train // batch_size
    n_test = X_test.shape[0]
    n_test_batches = n_test // batch_size
    center_pass = np.ones(len(center_nn))

    index = T.lscalar()
    cindex = T.lvector()
    x = sparse.csr_matrix('x')
    y = T.ivector('y')
    if useSVM:
        y1 = T.imatrix('y1')
    else:
        Y1_train = Y_train
        Y1_test = Y_test
        y1 = T.ivector('y1')

    #initialize
    old_all_loss = float('inf')
    passed_index = np.array([]) 
    passes = 0
    output_seq = ()

    train_index = np.arange(n_train)
    np.random.shuffle(train_index)
    train_index = train_index[:1000]

    if model is None:
        # launch the training loop
        print("Starting training...")
        for epoch in range(num_epochs):

            # Create neural network model
            print("Building model and compiling functions...")
            rng = np.random.RandomState(23455)
            network, params = build_cnn(x, input_dim=X_train.shape[1], num_classes=len(np.unique(Y_train)), useSVM=useSVM)
            cost = network.cost(y1)
            cost_vec = network.cost_vec(y1)
            
            if model is None:

                # create a list of gradients for all model parameters
                grads = T.grad(cost, params)

                updates = []
                for param_i, grad_i in zip(params, grads):
                    updates.append((param_i, param_i - learning_rate * grad_i))

                # train_model = theano.function([cindex], cost, updates=updates,
                #       givens={
                #         x: X_train[cindex,:],
                #         y1: Y1_train[cindex]})

                train_model = theano.function([x, y1], cost, updates=updates)

                # loss_model = theano.function([cindex], network.cost_vec(y1),
                #       givens={
                #         x: X_train[cindex,:],
                #         y1: Y1_train[cindex]})

                loss_model = theano.function([x, y1], cost_vec)

            # create a function to compute the mistakes that are made by the model
            # test_model = theano.function([cindex], network.errors(y),
            #          givens={
            #             x: X_test[cindex,:],
            #             y: Y_test[cindex]})

            test_model = theano.function([x, y, y1], [network.errors(y), cost])

            # error_model = theano.function([cindex], network.errors(y),
            #          givens={
            #             x: X_train[cindex,:],
            #             y: Y_train[cindex]})

            print("model complied.")

            start_time = time.time()

            #update model
            for iters in range(epoch_iters):
                start_index = 0
                train_loss = np.array([])
                while start_index < len(train_index):
                    end_index = min([start_index + batch_size, len(train_index)])
                    batch_index = train_index[start_index : end_index]
                    start_index = end_index
                    train_loss = np.append(train_loss, train_model(X_train[batch_index], Y1_train[batch_index]))
            passes += len(train_index) * epoch_iters
            passed_index = np.unique(np.append(passed_index, train_index))

            #compute loss for all samples
            start_index = 0
            loss_vec = np.array([])
            while start_index < n_train:
                end_index = min([start_index + batch_size, n_train])
                batch_index = range(start_index, end_index)
                loss_vec = np.append(loss_vec, loss_model(X_train[batch_index], Y1_train[batch_index]))
                start_index = end_index
            all_loss = np.mean(loss_vec)
            print('mean training loss = ', all_loss)

            if spld[0]: 

                #select sample: self-paced learning with diversity
                train_index = np.array([])
                for i in range(num_cluster):
                    iCluster = np.where(labels_ == i)[0]
                    iloss_vec = loss_vec[iCluster]
                    sortIndex = np.argsort(iloss_vec)
                    iloss_vec[sortIndex] -= spld[2] * np.divide(1,np.sqrt(range(1, 1+len(iCluster)))+np.sqrt(range(len(iCluster))))
                    K = min([spld[1], len(iloss_vec)-1])
                    train_index = np.append(train_index, iCluster[np.argpartition(iloss_vec, K)[:K]])
                train_index = train_index.astype('int32')

                spld[1] *= (1+spld[3])
                spld[1] = int(round(spld[1]))
                spld[2] *= (1+spld[4])

            else:

                #select sample: self-paced learning with diversity
                K = min([spld[1], n_train-1])
                train_index = np.argpartition(loss_vec, K)[:K]
                #increase learning pace
                spld[1] *= (1+spld[3])
                spld[1] = int(round(spld[1]))        

            #compute loss for all samples
            start_index = 0
            train_err = np.array([])
            train_loss = np.array([])
            while start_index < n_train:
                end_index = min([start_index + batch_size, n_train])
                batch_index = range(start_index, end_index)
                batch_err, batch_loss = test_model(X_train[batch_index], Y_train[batch_index], Y1_train[batch_index])
                train_err = np.append(train_err, batch_err)
                train_loss = np.append(train_loss, batch_loss)
                start_index = end_index
            this_train_err = np.mean(train_err)
            this_train_loss = np.mean(train_loss)

            start_index = 0
            test_err = np.array([])
            test_loss = np.array([])
            while start_index < n_test:
                end_index = min([start_index + batch_size, n_test])
                batch_index = range(start_index, end_index)
                batch_err, batch_loss = test_model(X_test[batch_index], Y_test[batch_index], Y1_test[batch_index])
                test_err = np.append(test_err, batch_err)
                test_loss = np.append(test_loss, batch_loss)
                start_index = end_index
            this_test_err = np.mean(test_err)
            this_test_loss = np.mean(test_loss)

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s, up to now {} trainings {} passes".format(
                epoch + 1, num_epochs, time.time() - start_time, len(passed_index), passes))
            print("  training err:\t\t{:.6f}".format(this_train_err*100.))
            print("  test err:\t\t{:.6f}".format(this_test_err*100.))
            print("  training loss:\t\t{:.2f}".format(this_train_loss))
            print("  test loss:\t\t{:.2f}".format(this_test_loss))

            output_seq = output_seq + (np.array([len(passed_index),passes,this_train_loss,this_test_loss,this_train_err*100.,this_test_err*100.]),)
            # increase curriculum rate
            if all_loss > 1.001 * old_all_loss:
                print('no improvement: reduce learning rate!')
                learning_rate *= 0.96
            old_all_loss = all_loss

        # dump the network weights to a file :
        with open('20newsgroups_shallow_model.pkl', 'wb') as f:
            pickle.dump(network, f)
        output_seq = np.vstack(output_seq)
        np.savetxt('20newsgroups_logistic_spl_25_3e-1_result.txt', output_seq)
    else:
        # load network weights from model file
        network = pickle.load(open('20newsgroups_shallow_model.pkl'))

    return output_seq


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on 20newsgroup using Theano.")
        print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N [MODEL]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['model'] = sys.argv[2]
        output_seq = main(**kwargs)

        plt.figure(figsize = (20, 10))
        plt.subplot(1,2,1)
        plt.plot(output_seq[:, 1], output_seq[:, 2], 'yo-', label = 'training loss')
        plt.plot(output_seq[:, 1], output_seq[:, 3], 'co-', label = 'training err')
        plt.plot(output_seq[:, 1], output_seq[:, 4], 'mo-', label = 'test err')
        plt.grid()
        plt.legend(fontsize='large', loc = 1)
        plt.ylabel('Error rate (%)')   
        plt.xlabel('Number of passed training samples (including copies)')

        plt.subplot(1,2,2)
        plt.plot(output_seq[:, 0], output_seq[:, 2], 'yo-', label = 'training loss')
        plt.plot(output_seq[:, 0], output_seq[:, 3], 'co-', label = 'training err')
        plt.plot(output_seq[:, 0], output_seq[:, 4], 'mo-', label = 'test err')
        plt.grid()
        plt.legend(fontsize='large', loc = 1)
        plt.ylabel('Error rate (%)') 
        plt.xlabel('Size of set of passed training samples')
       
        plt.savefig('20newsgroup_logistic_spl_25_3e-1.eps', format = 'eps', bbox_inches='tight')
        plt.show()