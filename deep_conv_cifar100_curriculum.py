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
from lasagne.layers import FlattenLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

def build_cnn(input_var, input_shape=(3, 32, 32),
              ccp_num_filters=[64, 128], ccp_filter_size=3,
              fc_num_units=[128, 128], num_classes=10,
              **junk):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None,) + input_shape,
                                        input_var=input_var)
    # conv-relu-conv-relu-pool layers
    for num_filters in ccp_num_filters:
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            #nonlinearity=lasagne.nonlinearities.rectify,
            nonlinearity=lasagne.nonlinearities.elu,
            #W=lasagne.init.GlorotUniform(gain='relu')
            W=lasagne.init.HeUniform(gain='relu')
            )
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            #nonlinearity=lasagne.nonlinearities.rectify,
            nonlinearity=lasagne.nonlinearities.elu,
            #W=lasagne.init.GlorotUniform(gain='relu')
            W=lasagne.init.HeUniform()
            )
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # fc-relu
    for num_units in fc_num_units:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=num_units,
            #nonlinearity=lasagne.nonlinearities.rectify,
            nonlinearity=lasagne.nonlinearities.elu,
            #W=lasagne.init.GlorotUniform(gain='relu')
            W=lasagne.init.HeUniform(gain='relu')
            )
    feanet = FlattenLayer(network)
    # output layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)
    return network, feanet

# ############################## Main program ################################

def main(num_epochs=60, model=None, 
        learning_rate=1e-2, momentum=0.9, decay_after_epochs = 3, loss_weight = 1.0e+7, curriculum_rate=0.05, 
        epoch_iters = 20, minibatch_size = 128, stain_factor = 60.0, num_cluster = 1000, 
        batch_size=4, k = 4, func = 'concavefeature', func_parameter = 0.5, spld = [False, 1.2, 2e-1, 4e-2, 4e-2]):
    # Check if cifar data exists
    if not os.path.exists("/home/tianyizhou/Downloads/cifar-100-python"):
        print("CIFAR-100 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = load_cifar100()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    #X_train_fea = data['X_fea']
    #labels_ = data['kmeans_label']
    #labels_weight = np.array([len(np.where(labels_==i)[0]) for i in np.unique(labels_)])
    #labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))

    #cluster_centers_ = data['kmeans_center']
    #center_nn = data['kmeans_center_nn']
    #center_nn = center_nn[:len(center_nn)/2]
    #num_cluster = cluster_centers_.shape[0] 
    n_train = len(Y_train)
    center_pass = np.ones(num_cluster)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    param = dict(ccp_num_filters=[64, 128], ccp_filter_size=3,
                fc_num_units=[256, 256], num_classes=100,
                learning_rate=1e-2, learning_rate_decay=0.5,
                momentum=0.9, momentum_decay=0.5,
                decay_after_epochs=10,
                batch_size=128, num_epochs=50)
    network, feanet = build_cnn(input_var, **param)
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
        momentum_var = theano.shared(lasagne.utils.floatX(momentum))
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=sh_lr, momentum=momentum_var)
        
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

    print("model complied, initialize curriculum...")

    #initialize
    #minGain, sinGain, optSubmodular = initSubmodularFunc(cluster_centers_, k)
    real_iter = 0
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

            if len(passed_index) <= n_train:

                if not spld[0]:

                    old_all_loss = 0
                    sum_center_pass = sum(center_pass)
                    center_pass_normalized = center_pass/sum_center_pass
                    stain_weight = np.power(center_pass_normalized, -1/stain_factor)
                    start_time = time.time()
                    # Update kmeans result and submodular function
                    if epoch % 15 == 0 and epoch <= 80:
                        train_fea = ()
                        for batch in iterate_minibatches0(X_train, minibatch_size):
                            train_fea = train_fea + (fea_fn(batch), )
                        train_fea = np.vstack(train_fea)
                        #labels_, cluster_centers_, center_nn = dataGroup(train_fea, Y_train, 0, num_cluster, 'cifar10', savefile = False)
                        labels_, cluster_centers_, center_nn = dataGroup0(train_fea, 0, num_cluster, 'cifar10', savefile = False)
                        labels_ = labels_.astype('int32')
                        center_nn = center_nn.astype('int32')
                        #print(np.histogram(labels_, bins=num_cluster))
                        labels_weight = np.array([len(np.where(labels_==i)[0]) for i in np.unique(labels_)])
                        labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))
                        minGain, sinGain, optSubmodular = initSubmodularFunc(cluster_centers_, k)
                    for iters in range(epoch_iters):

                        # compute loss
                        loss_vec_center = np.array([])
                        for batch in iterate_minibatches(X_train[center_nn], Y_train[center_nn], 200, shuffle=False, augment=False):
                            inputs, targets = batch
                            loss_vec_center = np.append(loss_vec_center, loss_fn(inputs, targets))
                        loss_vec_center *= labels_weight * stain_weight * (loss_weight / num_cluster)
                        all_loss = sum(loss_vec_center)
                        topkLoss = sum(np.partition(loss_vec_center, -k)[-k:])
                        if epoch % 4 == 0 and epoch > 120:
                            #topkIndex = np.argpartition(loss_vec_center, k)[:k]
                            #topkIndex = np.random.choice(num_cluster, k, replace=False, p=labels_weight/sum(labels_weight))
                            train_index = np.random.choice(n_train, 28, replace=False)
                        else:
                            #print(optSubmodular, topkLoss)                        
                            # update A (topkIndex)
                            optObj = optSubmodular + topkLoss
                            left_index = pruneGroundSet(minGain, sinGain, loss_vec_center, k)
                            topkIndex = modularLowerBound(cluster_centers_[left_index,:], k, func, func_parameter, loss_vec_center[left_index], optObj)
                            topkIndex = left_index[topkIndex]
                            center_pass[topkIndex] += 1.0
                            # update classifier (train_model)          
                            train_index = np.array([])
                            for i in range(len(topkIndex)):
                                train_index = np.append(train_index, np.where(labels_ == topkIndex[i])[0])
                            train_index = np.random.permutation(train_index.astype(int))

                        #print('number of training samples =', len(train_index))
                        passes += len(train_index)
                        passed_index = np.unique(np.append(passed_index, train_index))

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

                    start_time = time.time()
                    train_index = np.array([])
                    for i in range(num_cluster):

                        iCluster = np.where(labels_ == i)[0]
                        loss_vec_center = np.array([])
                        for batch in iterate_minibatches(X_train[iCluster], Y_train[iCluster], minibatch_size, shuffle=False, augment=False):
                            inputs, targets = batch
                            loss_vec_center = np.append(loss_vec_center, loss_fn(inputs, targets))
                        sortIndex = np.argsort(loss_vec_center)
                        thresh = spld[1] + spld[2] * np.divide(1, np.sqrt(np.arange(len(loss_vec_center))) + np.sqrt(np.arange(len(loss_vec_center))+1))
                        train_index = np.append(train_index, iCluster[sortIndex[np.less(loss_vec_center[sortIndex], thresh)]])

                    #print(train_index)
                    train_index = train_index.astype('int32')
                    for j in range(10):
                        train_err = 0
                        train_batches = 0
                        for batch in iterate_minibatches(X_train[train_index], Y_train[train_index], minibatch_size, shuffle=True, augment=True):
                            inputs, targets = batch
                            train_err += train_fn(inputs, targets)
                            train_batches += 1
                        passes += len(train_index)
                    passed_index = np.unique(np.append(passed_index, train_index))

            else:

                # shuffle training data
                #sh_lr.set_value(lasagne.utils.floatX(1e-1))
                train_index = np.arange(len(X_train))
                np.random.shuffle(train_index)
                train_index = train_index[:8000]
                passes += len(train_index)
                passed_index = np.unique(np.append(passed_index, train_index))
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
            loss_weight *= curriculum_rate + 1
            k = min([k + 8, num_cluster])
            spld[1] *= (1+spld[3])
            spld[2] *= (1+spld[4])

            # adjust learning rate as in paper
            if (epoch + 1) % decay_after_epochs == 0:
                sh_lr.set_value(
                    np.float32(sh_lr.get_value() * 0.95))
                momentum = (1.0 - (1.0 - momentum_var.get_value()) * 0.95) \
                           .clip(max=0.9999)
                momentum_var.set_value(lasagne.utils.floatX(momentum))

            if (epoch+1) == 41 or (epoch+1) == 71 or (epoch+1) == 101:
                # new_lr = sh_lr.get_value() * 0.95
                # print("New LR:"+str(new_lr))
                # sh_lr.set_value(lasagne.utils.floatX(new_lr))
                stain_factor -= 5

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
        numpy.savetxt('cifar100_convnet_cLearn_elu_k+8_result.txt', output_seq)

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
       
        plt.savefig('cifar100_convnet_cLearn_elu_k+8.eps', format = 'eps', bbox_inches='tight')
        plt.show()