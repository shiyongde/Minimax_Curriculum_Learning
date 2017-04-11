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
import theano
import theano.tensor as T
import lasagne
from cLearn_utils import *
import matplotlib.pyplot as plt

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in current working directory

# ##################### Build the neural network model #######################

#from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import FlattenLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

def build_cnn(input_var=None, n=5):
    
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n-3):
        l = residual_block(l)

    feanet = MaxPool2DLayer(l, pool_size=4)
    feanet = FlattenLayer(feanet)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    
    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
            l, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network, feanet

# ############################## Main program ################################

def main(n=5, num_epochs=82, model=None, 
        learning_rate=3e-2, loss_weight = 1.6e+6, curriculum_rate=0.08, 
        epoch_iters = 100, minibatch_size = 200, 
        batch_size=5, k = 5, num_cluster = 800, func = 'concavefeature', func_parameter = 0.5):
    # Check if cifar data exists
    if not os.path.exists("/home/tianyizhou/Downloads/cifar-10-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = load_cifar10_2()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    #X_train_fea = data['X_fea']
    labels_ = data['kmeans_label']
    labels_weight = np.array([len(np.where(labels_==i)[0]) for i in np.unique(labels_)])
    labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))

    cluster_centers_ = data['kmeans_center']
    center_nn = data['kmeans_center_nn']
    #num_cluster = cluster_centers_.shape[0]
    isize = int(np.sqrt(X_train.shape[-1]))  
    n_train = len(Y_train)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network, feanet = build_cnn(input_var, n)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    
    if model is None:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        outfea, prediction = lasagne.layers.get_output([feanet, network])
        loss_vec = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss_vec.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = loss + l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(network, trainable=True)
        sh_lr = theano.shared(lasagne.utils.floatX(learning_rate))
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=sh_lr, momentum=0.9)
        
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        loss_fn = theano.function([input_var, target_var], loss_vec)
        fea_fn = theano.function([input_var], outfea)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)


    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    #initialize
    #real_iter = 0
    #validation_frequency = 100
    old_epoch_all_loss = float('inf')
    loss_weight0 = loss_weight
    passed_index = np.array([]) 
    passes = 0
    output_seq = ()
    if model is None:
        # launch the training loop
        print("Starting training...")
        # train_err = 0
        # for batch in iterate_minibatches(X_train[center_nn], Y_train[center_nn], minibatch_size, shuffle=True, augment=True):
        #     inputs, targets = batch
        #     train_err += train_fn(inputs, targets)
        # We iterate over epochs:
        for epoch in range(num_epochs):

            if len(passed_index) <= n_train*0.3:

                old_all_loss = 0
                start_time = time.time()
                train_fea = ()
                for batch in iterate_minibatches0(X_train, minibatch_size):
                    train_fea = train_fea + (fea_fn(batch), )
                train_fea = np.vstack(train_fea)
                labels_, cluster_centers_, center_nn = dataGroup(train_fea, Y_train, 720, num_cluster, 'cifar10', savefile = False)
                labels_ = labels_.astype('int32')
                center_nn = center_nn.astype('int32')
                print(np.histogram(labels_, bins=num_cluster))
                labels_weight = np.array([len(np.where(labels_==i)[0]) for i in np.unique(labels_)])
                labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))
                minGain, sinGain, optSubmodular = initSubmodularFunc(cluster_centers_, k)

                for iters in range(epoch_iters):

                    # compute loss
                    loss_vec_center = np.array([])
                    for batch in iterate_minibatches(X_train[center_nn], Y_train[center_nn], minibatch_size, shuffle=False, augment=False):
                        inputs, targets = batch
                        loss_vec_center = np.append(loss_vec_center, loss_fn(inputs, targets))
                    loss_vec_center *= labels_weight*(loss_weight/num_cluster)
                    all_loss = sum(loss_vec_center)
                    topkLoss = sum(np.partition(loss_vec_center, -k)[-k:])
                    print(optSubmodular, topkLoss)
                    optObj = optSubmodular + topkLoss

                    # update A (topkIndex)
                    left_index = pruneGroundSet(minGain, sinGain, loss_vec_center, k)
                    topkIndex = modularLowerBound(cluster_centers_[left_index,:], k, func, func_parameter, loss_vec_center[left_index], optObj)
                    topkIndex = left_index[topkIndex]

                    # update classifier (train_model)          
                    train_index = np.array([])
                    for i in range(len(topkIndex)):
                        train_index = np.append(train_index, np.where(labels_ == topkIndex[i])[0])
                    train_index = np.random.permutation(train_index.astype(int))
                    print('number of training samples =', len(train_index))
                    passes += len(train_index)
                    passed_index = np.unique(np.append(passed_index, train_index))

                    # training by mini-batch sgd
                    # print("update model...")
                    # raw_input("Press Enter to continue...")
                    # start_index = 0
                    # train_loss = np.array([])
                    # while start_index < len(train_index):
                    #     end_index = min([start_index + minibatch_size, len(train_index)])
                    #     batch_index = train_index[start_index : end_index]
                    #     start_index += end_index
                    #     train_loss = np.append(train_loss, train_fn(X_train[batch_index], Y_train[batch_index]))
                    # this_train_loss = np.mean(train_loss)    

                    #update model
                    #for j in range(2):
                    train_err = 0
                    train_batches = 0
                    for batch in iterate_minibatches(X_train[train_index], Y_train[train_index], minibatch_size, shuffle=True, augment=True):
                        inputs, targets = batch
                        train_err += train_fn(inputs, targets)
                        train_batches += 1
                    #train_err = train_err/train_batches

            else:

                # shuffle training data
                sh_lr.set_value(lasagne.utils.floatX(1e-1))
                train_index = np.arange(100000)
                np.random.shuffle(train_index)
                # X_train = X_train[train_indices,:,:,:]
                # Y_train = Y_train[train_indices]

                # In each epoch, we do a full pass over the training data:
                start_time = time.time()
            
                #update model
                train_err = 0
                train_batches = 0
                for batch in iterate_minibatches(X_train[train_index], Y_train[train_index], minibatch_size, shuffle=True, augment=True):
                    inputs, targets = batch
                    train_err += train_fn(inputs, targets)
                    train_batches += 1
                #train_err = train_err/train_batches

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            train_err = 0
            train_batches = 0
            for batch in iterate_minibatches(X_train, Y_train, 500, shuffle=False, augment=False):
                inputs, targets = batch
                #train_err += loss_fn(inputs, targets)
                train_err_vec = loss_fn(inputs, targets)
                train_err += np.mean(train_err_vec)
                train_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s, up to now {} trainings {} passes".format(
                epoch + 1, num_epochs, time.time() - start_time, len(passed_index), passes))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            output_seq = output_seq + (numpy.array([len(passed_index),passes,train_err / train_batches,val_err / val_batches,val_acc / val_batches * 100.]),)
            # increase curriculum rate
            loss_weight *= (curriculum_rate + 1)

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch+1) == 41 or (epoch+1) == 61:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        np.savez('cifar10_deep_residual_model.npz', *lasagne.layers.get_all_param_values(network))
        output_seq = numpy.vstack(output_seq)
        np.savetxt('cifar10_residual_curriculum_result.txt', output_seq)
    else:
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # Calculate validation error of model:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    return output_seq


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
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
        plt.plot(output_seq[:, 1], output_seq[:, 3], 'co-', label = 'validation loss')
        plt.plot(output_seq[:, 1], output_seq[:, 4], 'mo-', label = 'validation accuracy')
        plt.grid()
        plt.legend(fontsize='large', loc = 1)
        plt.ylabel('Error rate (%)')   
        plt.xlabel('Number of passed training samples (including copies)')

        plt.subplot(1,2,2)
        plt.plot(output_seq[:, 0], output_seq[:, 2], 'yo-', label = 'training loss')
        plt.plot(output_seq[:, 0], output_seq[:, 3], 'co-', label = 'validation loss')
        plt.plot(output_seq[:, 0], output_seq[:, 4], 'mo-', label = 'validation accuracy')
        plt.grid()
        plt.legend(fontsize='large', loc = 1)
        plt.ylabel('Error rate (%)') 
        plt.xlabel('Size of set of passed training samples')
       
        plt.savefig('clearn_cifar10_residual.eps', format = 'eps', bbox_inches='tight')
        plt.show()