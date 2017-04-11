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

import theano
import theano.tensor as T

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


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

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

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
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
            (test_set_x, test_set_y)]
    return rval

def update_permutation(score, perm, k, f):

    topk_index = numpy.argsort(score)[-k:][::-1]
    perm = numpy.array(list(set(range(len(score))) - set(topk_index)))
    return 

def modular_lower_bound(X, k, offset, func = 'concavefeature', func_parameter = 0.5, perm = [], init_method = 'greedy'):

    if len(perm) == 0:
        if init_method == 'random':
            perm = numpy.random.permutation(X.shape[0])
            if func == 'facloc':
                f = facloc_graph(X, func_parameter[0], func_parameter[1])
            elif func == 'satcoverage':
                f = satcoverage(X, func_parameter[0], func_parameter[1])    
            elif func == 'concavefeature':
                f = concavefeature(X, func_parameter)
            elif func == 'setcover':
                f = setcover(X, func_parameter[0], func_parameter[1])
            elif func == 'videofeaturefunc':
                f = videofeaturefunc(X)
                X = X[1]
            else:
                print('Function can only be facloc, satcoverage, concavefeature, setcover or videofeaturefunc')

            print('finish building submodular function for data size', X.shape[0], '\n')
        elif init_method == 'greedy':
            g = greedy(X, func = func, func_parameter = func_parameter, save_memory = [False, 8], offset = offset)
            perm, f_obj, f_Vsize = g(k)
            f = g.f
    else:
        f = func

    if len(perm) <= X.shape[0]:
        perm = numpy.append(perm, numpy.random.permutation(list(set(range(X.shape[0])) - set(perm))))
    else:
        print('length of perm should be less than', X.shape[0])

    print('computing subdifferentials')
    score = numpy.zeros(X.shape[0])
    nn, score[0] = f.evaluate([perm[0]])
    for i in range(1, X.shape[0]):
        nn, score[i] = f.evaluate_incremental(nn, perm[i])
    score[1:] = score[1:] - score[:-1]

    return score,perm,f

def sgd_optimization_mnist(learning_rate=0.1, n_curriculum_epochs=100,
                           dataset='mnist.pkl.gz',
                           batch_size=5, curriculum_rate=0.3, loss_weight = 3.2e+4, k = 5, num_cluster = 800):
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
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_train // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    cindex = T.lvector()  # index to a [mini]batch


    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
    cost_vec = classifier.negative_log_likelihood_vec(y)

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

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[cindex],
        outputs=cost,
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
    # end-snippet-3

    print('clustering...')
    #kmeans = KMeans(n_clusters = num_cluster).fit(train_set_x.get_value(borrow=True))
    #numpy.savetxt('mnist_kmeans_labels.txt', kmeans.labels_)
    #numpy.savetxt('mnist_kmeans_centers.txt', kmeans.cluster_centers_)
    labels_ = numpy.loadtxt('mnist_kmeans_labels.txt').astype(int)
    cluster_centers_ = numpy.loadtxt('mnist_kmeans_centers.txt')


    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
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
    loss_vec = loss_model(range(n_train)) * loss_weight / n_train
    loss_vec_center = numpy.asarray([sum(loss_vec[labels_ == i]) for i in range(num_cluster)])
    score, perm, sf = modular_lower_bound(cluster_centers_, k, loss_vec_center)
    n_iter = 10
    converge = 1e-4
    real_iter = 0
    validation_frequency = 10
    loss_weight0 = loss_weight
    passed_index = numpy.array([])
    for curriculum_epoch in range(n_curriculum_epochs):

        if curriculum_epoch > 11:
            learning_rate = 0.12

        #initialize
        print('Epoch', curriculum_epoch)
        train_loss = -1
        curriculum_score = 0
        old_all_loss = 0
        for iters in range(n_iter):

            loss_vec = loss_model(range(n_train)) * loss_weight / n_train
            #print('mean(loss), mean(score)', numpy.mean(loss_vec), numpy.mean(score))
            loss_vec_center = numpy.asarray([sum(loss_vec[labels_ == i]) for i in range(num_cluster)])
            all_loss = sum(loss_vec)

            if curriculum_epoch <= 11:
                score_vec = loss_vec_center
                score_vec[perm] += score
                new_curriculum_index = numpy.argsort(score_vec)[-k:][::-1]

                # test if objective increases
                if train_loss != -1 and len(set(curriculum_index) - set(new_curriculum_index)) != 0:
                    new_curriculum_score = sf.evaluate(new_curriculum_index)[1]
                    if new_curriculum_score + numpy.sum(loss_vec_center[new_curriculum_index]) > curriculum_score + numpy.sum(loss_vec_center[curriculum_index]):
                        curriculum_index = new_curriculum_index
                        curriculum_score = new_curriculum_score
                        train_loss = -1
                    else:
                        new_curriculum_index = curriculum_index

                # compute loss
                new_train_loss = numpy.sum(loss_vec_center[new_curriculum_index])
                if train_loss != -1 and len(set(curriculum_index) - set(new_curriculum_index)) == 0 and new_train_loss >= train_loss:
                    score, perm, sf = modular_lower_bound(cluster_centers_, k, loss_vec_center, func = sf, perm = curriculum_index)
                    train_loss = -1
                else:
                    train_loss = new_train_loss
                    curriculum_index = new_curriculum_index

            else:
                curriculum_index = numpy.random.permutation(num_cluster)[:k]

            train_index = numpy.array([])
            for i in range(len(curriculum_index)):
                train_index = numpy.append(train_index, numpy.where(labels_ == curriculum_index[i])[0])
            train_index = train_index.astype(int)
            print('number of training samples =', len(train_index))
            passed_index = numpy.unique(numpy.append(passed_index, train_index))

            for i in range(2):
                curriculum_avg_cost = train_model(train_index)

            diff_loss = old_all_loss - all_loss
            if  diff_loss >= 0 and diff_loss <= all_loss * converge:
                real_iter = iters
                break
            else:
                old_all_loss = all_loss
                if (iters + real_iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print(
                        'minibatch %i, validation error %f %%' %
                        (
                            iters + real_iter + 1,
                            this_validation_loss * 100.
                        )
                    )
                    print('Up to now %i training samples are used'%(len(passed_index)))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, (iters + real_iter + 1) * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i)
                                       for i in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(
                            (
                                'minibatch %i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                iters + real_iter + 1,
                                test_score * 100.
                            )
                        )

                        # save the best model
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(classifier, f)

        #if curriculum_epoch % 10 == 0:
            #loss_weight = loss_weight0
        learning_rate *= 0.92
        #else:
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
    sgd_optimization_mnist()
