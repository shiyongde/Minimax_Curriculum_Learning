import argparse
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import cosine_distances
#from scipy.spatial.distance import cdist
from scipy.sparse import issparse
#from sklearn.manifold import TSNE

# from models import *

#from facloc_graph import facloc_graph
#from satcoverage import satcoverage
#from concavefeature import concavefeature
#from setcover import setcover
from submdl2D import submdl_teach_welfare
from greedy2D import greedy2D
#from randomGreedy import randomGreedy

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch LeNet Training')
parser.add_argument('--epochs', default=31, type=int,
                    help='number of total epochs to run for learner')
parser.add_argument('--epochs4loss', default=10, type=int,
                    help='number of total epochs to run for loss predictor')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=8e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_lp', '--learning-rate-loss-predict', default=5e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=2, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--lr-freq', default=8, type=int,
                    help='learning rate changing frequency (default: 5)')
parser.add_argument('--assign-freq', default=1, type=int,
                    help='training set assignment changing frequency (default: 1)')
parser.add_argument('--random-freq', default=10, type=int,
                    help='insert epoch with random samples frequency (default: 10)')
parser.add_argument('--fea-freq', default=100, type=int,
                    help='insert epoch with random samples frequency (default: 10)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--name', default='LeNet5', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--num_cluster', default=800, type=int,
                    help='Number of clusters (default: 1000)')
parser.add_argument('--num_learner_per_cluster', default=1, type=int,
                    help='Initial number of learners assigned to each cluster (default: 4)')
parser.add_argument('--num_cluster_per_learner', default=30, type=int,
                    help='Initial number of clusters assigned to each learner (default: 100)')
parser.add_argument('--deltak', default=1, type=int,
                    help='decreased number of clusters in training set per epoch (default: 1)')
parser.add_argument('--loss_weight', default=2.8e+1, type=float,
                    help='Initial weight of loss term (default: 1.5e+9)')
parser.add_argument('--curriculum_rate', default=0.9, type=float,
                    help='Increasing ratio of loss weight (default: 0.03)')
parser.add_argument('--num_learner', default=8, type=int,
                    help='number of learners/models (default: 10)')
parser.add_argument('--func', default='facloc', type=str,
                    help='Submodular function for diversity regularization (default: facloc)')
parser.add_argument('--func_parameter', default=['euclidean_gaussian', 20], type=float,
                    help='Parameter of submodular function (default: 0.5)')
parser.add_argument('--use_submodular', default=0, type = int,
                    help='Parameter of submodular function (default: 0.5)')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

## network
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc2(x))
        x = F.relu(self.fc3(y))
        return y, x
    def name(self):
        return 'mlpnet'

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x) # F.sigmoid(x)
        x = self.bn2(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        y = self.bn3(self.fc1(x))
        x = F.relu(y)
        x = self.fc2(x)
        return y, x
    def name(self):
        return 'lenet'

class LeNet_loss_predict(nn.Module):
    def __init__(self, nout):
        super(LeNet_loss_predict, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        init.xavier_uniform(self.conv1.weight.data)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        init.xavier_uniform(self.conv2.weight.data)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(4*4*16, 100)
        init.xavier_uniform(self.fc1.weight.data)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, nout)
        init.xavier_uniform(self.fc2.weight.data)
        self.bn4 = nn.BatchNorm1d(nout)
        # self.fc3 = nn.Linear(40, nout)
        # init.xavier_uniform(self.fc3.weight.data)
        self.ceriation = nn.MSELoss()
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.elu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.bn2(self.conv2(x))
        x = F.elu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*16)
        x = self.bn3(self.fc1(x))
        x = F.elu(x)
        # x = self.bn4(self.fc2(x))
        # x = F.log_softmax(F.elu(x))
        x = F.elu(self.fc2(x))
        # x = self.bn4(self.fc2(x))
        # x = F.sigmoid(x)
        # x = self.fc3(x)
        return x
    def name(self):
        return 'lenet_loss_predict'

def main():

    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))

    # data preprocessing
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # train/test set
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_set = datasets.MNIST('/mnt/s0/tianyizh/data', train=True, download=True, transform=trans)
    train_set_copy = datasets.MNIST('/mnt/s0/tianyizh/data', train=True, transform=trans)
    val_set = datasets.MNIST('/mnt/s0/tianyizh/data', train=False, transform=trans)
    val_set_copy = datasets.MNIST('/mnt/s0/tianyizh/data', train=False, transform=trans)
    
    train_label = train_set.train_labels
    n_train = len(train_label)

    fea_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    # learner models
    model = []
    for i in range(args.num_learner):
        model.append(LeNet().cuda())

    for i in range(args.num_learner):
        # get the number of model parameters
        print('model['+str(i)+']: number of parameters: {}'.format(
            sum([p.data.nelement() for p in model[i].parameters()])))
        # model[i] = model[i].cuda()

    # loss predictor model
    # net = LeNet_loss_predict(args.num_learner).cuda()

    # for training on multiple GPUs.
    CUDA_VISIBLE_DEVICES=2,3
    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # loss function for learner model
    criterion = nn.CrossEntropyLoss().cuda()
    # loss function for loss predictor
    net_loss = nn.MSELoss().cuda()
    # net_loss = nn.KLDivLoss().cuda()
    # per-sample loss function
    logsoftmax = nn.LogSoftmax().cuda()

    # optimizer for learner models
    optimizer = []
    for i in range(args.num_learner):
        optimizer.append(torch.optim.SGD(model[i].parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay))
    # optimizer for loss predictor
    # optimizer_net = torch.optim.Adam(net.parameters(), args.lr_lp, weight_decay=args.weight_decay)

    # initialization
    passed_index = [np.array([]).astype('int32') for i in range(args.num_learner)]
    passes = np.zeros(args.num_learner).astype('int32')
    output_seq = ()
    submodular_time = 0
    training_time = 0
    logfile = open('mnist_LeNet_ECL_l2_800_wok_wosub_log.txt','a')

    # compute loss and feature of training samples
    _, train_fea = loss_loader(fea_loader, model, logsoftmax, True)

    # clustering
    labels_, labels_weight, cluster_centers_, center_nn = get_cluster(train_fea, 80, args.num_cluster, 'mnist', savefile = False)
    args.num_cluster = len(center_nn)

    # Initialize submodular function and greedy algorithm
    if args.use_submodular:
        rewardMat = np.zeros((args.num_learner, args.num_cluster))
        SubmodularF = submdl_teach_welfare(cluster_centers_, rewardMat, args.func, args.func_parameter)
        greedyAlg = greedy2D(SubmodularF)
        avg_train_size = (args.num_learner_per_cluster * args.num_cluster) / args.num_learner
        topk_F = args.num_learner * sum(np.partition(greedyAlg.sinGain, -avg_train_size)[-avg_train_size:])

    end = time.time()
    epoch_end = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # training set assignment ---------------------------------------------------------------------------------
        # insert epoch of random samples
        if epoch % args.random_freq == 0 and epoch >= 200:
            train_subset = []
            for i in range(args.num_learner):
                train_subset.append(np.random.choice(n_train, args.batch_size*3, replace=False).tolist())
                passed_index[i] = np.unique(np.append(passed_index[i], train_subset[i]))

        # submodular teacher welfare to select samples
        elif epoch % args.assign_freq == 0:                           
            submodular_start_time = time.time()

            # compute loss of cluster centroids and transform to reward
            csampler = torch.utils.data.sampler.SubsetRandomSampler(center_nn)
            center_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=csampler, **kwargs)
            lossMat, lossAvg, PrecAvg = loss_loader(center_loader, model, logsoftmax, False)
            # print 'PrecAvg of centers', PrecAvg

            # submodular optimization to select clusters for each learner
            if args.use_submodular:
                # rewardMat = (lossMat.max(1)[0].expand(lossMat.size(0), lossMat.size(1)) - lossMat).numpy().T
                rewardMat = ((4.0 - lossMat).numpy().T) * (labels_weight * args.loss_weight)
                train_clusters, greedyObj, Vsize = greedyAlg(args.num_learner_per_cluster, rewardMat)
                topk_L = sum([sum(np.partition(rewardMat[:, i], -args.num_learner_per_cluster)[-args.num_learner_per_cluster:]) for i in range(args.num_cluster)])
                print 'greedyObj, topk_L, topk_F:', greedyObj, topk_L, topk_F
            else:
                topk_index = lossMat.topk(args.num_learner_per_cluster, dim=1, largest=False)[1].numpy()
                train_clusters = [[] for i in range(args.num_learner)]
                for i in range(args.num_cluster):
                    for j in topk_index[i]:
                        train_clusters[j].append(i)
            # select partial data (with large loss) for training
            if args.num_cluster_per_learner > 0:
                for i in range(args.num_learner):
                    num_cluster_i = min([args.num_cluster_per_learner, len(train_clusters[i])])
                    train_clusters[i] = np.array(train_clusters[i])[np.argpartition(lossMat[:,i].numpy()[train_clusters[i]], -num_cluster_i)[-num_cluster_i:]].tolist()
            train_size = np.array(map(len, train_clusters))
            # print 'number of clusters assigned to each learner:', train_size
            active_learner_set = np.where(train_size > 0)[0]

            # transform from selected clusters to training sample index
            train_subset = [[] for i in range(args.num_learner)]
            for i in active_learner_set:
                train_subset[i] = np.concatenate([labels_[j] for j in train_clusters[i]]).tolist()
                passed_index[i] = np.unique(np.append(passed_index[i], train_subset[i]))

            submodular_time += (time.time() - submodular_start_time)
        # training set assignment ---------------------------------------------------------------------------------

        # training stage ------------------------------------------------------------------------------------------
        print('--------------------Now update learner models--------------------')
        training_start_time = time.time()
        loss_iter = np.zeros(args.num_learner)
        for i in active_learner_set:
            # change learning rate
            # print('--------------------Now update model['+str(i)+']--------------------')

            # load training set for each model
            csampler = torch.utils.data.sampler.SubsetRandomSampler(train_subset[i])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=csampler, **kwargs)

            # train learner i
            loss_iter[i] = train(train_loader, model[i], criterion, optimizer[i], epoch, True)

            # record passes
            passes[i] += len(train_subset[i])

        training_time += (time.time() - training_start_time)
        # training stage ------------------------------------------------------------------------------------------

        # update learning rate and learning pace
        if epoch % args.lr_freq == 0:
            for i in range(args.num_learner):
                adjust_learning_rate(optimizer[i], epoch)
            args.loss_weight *= args.curriculum_rate + 1
            args.num_learner_per_cluster = max([args.num_learner_per_cluster - args.deltak, 1])
            print 'loss_weight, num_learner_per_cluster', args.loss_weight, args.num_learner_per_cluster

        # update clustering and submodular function F
        if epoch % args.fea_freq == 0:

            # clustering
            labels_, labels_weight, cluster_centers_, center_nn = get_cluster(train_fea, 80, args.num_cluster, 'mnist', savefile = False)
            args.num_cluster = len(center_nn)

            # Initialize submodular function and greedy algorithm
            if args.use_submodular:
                rewardMat = np.zeros((args.num_learner, args.num_cluster))
                SubmodularF = submdl_teach_welfare(cluster_centers_, rewardMat, args.func, args.func_parameter)
                greedyAlg = greedy2D(SubmodularF)
                avg_train_size = (args.num_learner_per_cluster * args.num_cluster) / args.num_learner
                topk_F = args.num_learner * sum(np.partition(greedyAlg.sinGain, -avg_train_size)[-avg_train_size:])

        # save and print intermediate results
        if epoch % args.print_freq == 0:

            # train loss predictor ----------------------------------------------------------------------------------
            train_loss, train_fea = loss_loader(fea_loader, model, logsoftmax, True)
            train_set_copy.train_labels = train_loss
            # print 'max, min, mean', train_loss.max(0)[0], train_loss.max(1)[0], train_loss.min(0)[0], train_loss.min(1)[0], train_loss.mean(0)[0], train_loss.mean(1)[0]
            train_loss_loader = torch.utils.data.DataLoader(train_set_copy, batch_size=args.batch_size, shuffle=True, **kwargs)
            print('--------------------Now update loss predictor--------------------')
            regression_loss = np.zeros(args.epochs4loss)
            net = LeNet_loss_predict(args.num_learner).cuda()
            print('loss predictor: number of parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
            optimizer_net = torch.optim.Adam(net.parameters(), args.lr_lp, weight_decay=args.weight_decay)
            for epoch_lp in range(args.epochs4loss):
                regression_loss[epoch_lp] = train(train_loss_loader, net, net_loss, optimizer_net, epoch_lp, False)
                if epoch % args.lr_freq == 0:
                    adjust_learning_rate(optimizer_net, epoch_lp)
                print 'loss predictor training loss: ', regression_loss[epoch_lp]

            # for test of loss predictor
            val_loss, _ = loss_loader(val_loader, model, logsoftmax, True)
            val_set_copy.test_labels = val_loss
            val_loss_loader = torch.utils.data.DataLoader(val_set_copy, batch_size=args.batch_size, shuffle=True, **kwargs)
            regression_loss_val = validate(val_loss_loader, net, net_loss, epoch_lp)
            print 'loss predictor validation loss: ', regression_loss_val
            # train loss predictor ----------------------------------------------------------------------------------

            # evaluate on validation set
            losses_val, prec1_val, losses_learner_val, prec1_learner_val, _ = validate_ensemble(val_loader, model, net, criterion)
            # evaluate on training set
            losses_train, prec1_train, losses_learner_train, prec1_learner_train, _ = validate_ensemble(fea_loader, model, net, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1_val[-1] > best_prec1
            best_prec1 = max(prec1_val[-1], best_prec1)

            start = time.time()
            epoch_time = start - epoch_end
            epoch_end = time.time()
            total_time = start - end
            unique_passes = np.array([len(passed_index[i]) for i in range(args.num_learner)])
            total_unique_passes = len(np.unique(np.concatenate(passed_index)))
            output_seq = output_seq + (np.concatenate((passes, unique_passes, losses_train, prec1_train,losses_learner_train, prec1_learner_train,losses_val, prec1_val,losses_learner_val,prec1_learner_val,[args.num_learner_per_cluster,args.loss_weight])),)

            print '--------------------training set info--------------------'
            print 'passes', passes
            print 'unique_passes', np.array([len(passed_index[i]) for i in range(args.num_learner)])

            print '--------------------result per learner--------------------'
            print 'losses_learner_train', np.around(np.array(losses_learner_train), decimals = 3)
            print 'prec1_learner_train', np.around(np.array(prec1_learner_train), decimals = 2)
            print 'losses_learner_val', np.around(np.array(losses_learner_val), decimals = 3)
            print 'prec1_learner_val', np.around(np.array(prec1_learner_val), decimals = 2)

            print '--------------------result of ensemble--------------------'
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {epoch_time:.3f} ({total_time:.3f})\n'
                    'Loss {loss_train[0]:.4f} ({loss_test[0]:.4f})\t'
                    'Loss {loss_train[1]:.4f} ({loss_test[1]:.4f})\t'
                    'Loss {loss_train[2]:.4f} ({loss_test[2]:.4f})\t'
                    'Loss {loss_train[3]:.4f} ({loss_test[3]:.4f})\n'
                    'Prec@1 {prec_train[0]:.4f} ({prec_test[0]:.4f})\t'
                    'Prec@1 {prec_train[1]:.4f} ({prec_test[1]:.4f})\t'
                    'Prec@1 {prec_train[2]:.4f} ({prec_test[2]:.4f})\t'
                    'Prec@1 {prec_train[3]:.4f} ({prec_test[3]:.4f})\t'.format(
                        epoch, total_unique_passes, np.sum(passes),
                        epoch_time=epoch_time, total_time=total_time, 
                        loss_test=losses_val, loss_train=losses_train,
                        prec_test=prec1_val, prec_train=prec1_train))


            logfile.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {epoch_time:.3f} ({total_time:.3f})\n'
                    'Loss {loss_train[0]:.4f} ({loss_test[0]:.4f})\t'
                    'Loss {loss_train[1]:.4f} ({loss_test[1]:.4f})\t'
                    'Loss {loss_train[2]:.4f} ({loss_test[2]:.4f})\t'
                    'Loss {loss_train[3]:.4f} ({loss_test[3]:.4f})\n'
                    'Prec@1 {prec_train[0]:.4f} ({prec_test[0]:.4f})\t'
                    'Prec@1 {prec_train[1]:.4f} ({prec_test[1]:.4f})\t'
                    'Prec@1 {prec_train[2]:.4f} ({prec_test[2]:.4f})\t'
                    'Prec@1 {prec_train[3]:.4f} ({prec_test[3]:.4f})\t'.format(
                        epoch, total_unique_passes, np.sum(passes), 
                        epoch_time=epoch_time, total_time=total_time, 
                        loss_test=losses_val, loss_train=losses_train,
                        prec_test=prec1_val, prec_train=prec1_train)+'\n')
    
    # save result to file
    output_seq = np.vstack(output_seq)
    np.savetxt('mnist_LeNet_ECL_l2_800_wok_wosub_result.txt', output_seq)
    logfile.close()

    # show final result
    print 'Best accuracy: ', best_prec1
    print 'SubmodularMax time: ', submodular_time
    print 'Training time: ', training_time
    print 'Total time: ', time.time() - end

def train(train_loader, model, criterion, optimizer, epoch, learner=True):
    """Train for one epoch on the training set"""
    # batch_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()

    # end = time.time()
    output_pred = ()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        if learner:
            _, output = model(input_var)
            loss = criterion(output, target_var)
        else:
            output = model(input_var)
            loss = criterion(output, target_var)
            #print 'output', output.data
        # output_pred = output_pred + (output.data, )


        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # output_pred = torch.stack(output_pred, 0)
    # np.save('output_pred', output_pred)

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

    # log to TensorBoard
    # if args.tensorboard:
        # log_value('train_subset_loss', losses.avg, epoch)
        # log_value('train_subset_acc', top1.avg, epoch)

    return losses.avg

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg

def loss_loader(train_loader, model, criterion, loss_predictor=True):

    # For learner models: switch to evaluate mode
    num_model = len(model)
    losses = []
    top1 = []
    for i in range(num_model):
        model[i].eval()
        losses.append(AverageMeter())
        top1.append(AverageMeter())

    # generate regression target for loss predictor
    train_loss = ()
    train_fea = ()
    # kwargs = {'num_workers': 1, 'pin_memory': True}
    for batch_index, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output and confidence
        alloss = ()
        outfea = ()
        for i in range(num_model):

            # compute per-sample loss and output feature
            modeli = model[i]
            if loss_predictor:
                outputf, outputi = modeli(input_var)
                outfea = outfea + (outputf,)
            else:
                _, outputi = modeli(input_var)
            lossi = -criterion(outputi).data.gather(1, target_var.data.view(-1,1))
            alloss = alloss + (lossi, )

            # measure accuracy and record loss
            if not loss_predictor:
                prec1 = accuracy(outputi.data, target, topk=(1,))[0]
                losses[i].update(lossi.mean(), input.size(0))
                top1[i].update(prec1[0], input.size(0))

        train_loss = train_loss + (torch.cat(alloss, 1), )
        if loss_predictor:
            train_fea = train_fea + (torch.stack(outfea,0).sum(0).squeeze(), )
        # print 'train_fea_batch shape:', train_fea_batch.size()

    # preprocess of losses as target (clamp, -log)

    train_loss = torch.cat(train_loss, 0).cpu()
    # np.save('train_loss', train_loss.cpu().numpy())
    # sys.exit(1)
    train_loss = train_loss.clamp(1e-12, 4.0)
    if loss_predictor:
        # train_loss = torch.div(train_loss, train_loss.sum(1).expand(train_loss.size(0), train_loss.size(1)))
        train_loss = -train_loss.log()
        # _, ind_loss = torch.topk(train_loss, args.num_learner_per_cluster, 1, largest = False)
        # train_loss = torch.zeros(train_loss.size()).scatter_(1, ind_loss, 1.0/args.num_learner_per_cluster)
        train_fea = torch.cat(train_fea, 0).data.cpu().numpy()
    else:
        train_loss = train_loss.clamp(1e-12, 4.0)
        lossAvg = [losses[i].avg for i in range(num_model)]
        PrecAvg = [top1[i].avg for i in range(num_model)]

    return (train_loss, train_fea) if loss_predictor else (train_loss, lossAvg, PrecAvg)

def validate_ensemble(val_loader, model, net, criterion, test=True):
    """Perform validation on the validation set"""
    if test:
        set_name = 'Test'
    else:
        set_name = 'Train'
    # output_feature = ()

    batch_time = AverageMeter()
    losses = []
    top1 = []
    for i in range(4):
        losses.append(AverageMeter())
        top1.append(AverageMeter())

    # define loss
    confidence = nn.LogSoftmax().cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    nllloss = nn.NLLLoss().cuda()

    # switch to evaluate mode
    # initialization
    num_model = len(model)
    losses_learner = []
    top1_learner = []
    for i in range(num_model):
        model[i].eval()
        losses_learner.append(AverageMeter())
        top1_learner.append(AverageMeter())
    net.eval()

    end = time.time()
    for batch_index, (input, target) in enumerate(val_loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output and confidence
        output = ()
        confid = ()
        alloss = ()
        for i in range(num_model):
            _, outputi = model[i](input_var)
            output = output + (outputi, )
            confidi = confidence(outputi)
            confid = confid + (confidi, )
            alloss = alloss + (-confidi.gather(1, target_var.view(-1,1)), )

        # num_model x batch_size x num_class
        output = torch.stack(output, 0)
        confid = torch.stack(confid, 0)
        batch_size = confid.size(1)
        num_class = confid.size(2)
        # num_model x batch_size
        confid_max = torch.max(confid, 2)[0].view(num_model, batch_size)
        alloss = torch.cat(alloss, 1).transpose(1,0)
        # print 'alloss size', alloss.size()

        # measure accuracy and record loss of different ensemble methods

        # topk confidence (greedy)
        val_confid, ind_confid = torch.topk(confid_max, args.num_learner_per_cluster, 0)
        # print 'size of output and ind_confid', output.size(), ind_confid.size()
        output_ensemble = output.gather(0, ind_confid.repeat(num_class, 1, 1).permute(1,2,0)).mean(0).squeeze(0)
        loss = criterion(output_ensemble, target_var)
        losses[0].update(loss.data[0], batch_size)
        prec1 = accuracy(output_ensemble.data, target, topk=(1,))[0]
        top1[0].update(prec1[0], batch_size)

        # topk smallest loss (oracle)
        val_loss, ind_loss = torch.topk(alloss, args.num_learner_per_cluster, 0, largest = False)
        # print 'size of output and ind_loss', output.size(), ind_loss.size()
        output_ensemble = output.gather(0, ind_loss.repeat(num_class, 1, 1).permute(1,2,0)).mean(0).squeeze(0)
        # output_ensemble = output.gather(0, ind_loss).mean(0).squeeze(0)
        loss = criterion(output_ensemble, target_var)
        losses[1].update(loss.data[0], batch_size)
        prec1 = accuracy(output_ensemble.data, target, topk=(1,))[0]
        top1[1].update(prec1[0], batch_size)

        # average
        output_ensemble = output.mean(0).squeeze(0)
        loss = criterion(output_ensemble, target_var)
        losses[2].update(loss.data[0], batch_size)
        prec1 = accuracy(output_ensemble.data, target, topk=(1,))[0]
        top1[2].update(prec1[0], batch_size)

        # topk smallest predicted loss (ours)
        val_ours, ind_ours = torch.topk(net(input_var).transpose(1,0), args.num_learner_per_cluster, 0)
        # print 'size of output and ind_ours', output.size(), ind_ours.size()
        output_ensemble = output.gather(0, ind_ours.repeat(num_class, 1, 1).permute(1,2,0)).mean(0).squeeze(0)
        # output_ensemble = output.gather(0, ind_ours).mean(0).squeeze(0)
        loss = criterion(output_ensemble, target_var)
        losses[3].update(loss.data[0], batch_size)
        prec1 = accuracy(output_ensemble.data, target, topk=(1,))[0]
        top1[3].update(prec1[0], batch_size)        

        # validation per model
        for i in range(args.num_learner):
            loss = criterion(output[i], target_var)
            losses_learner[i].update(loss.data[0], batch_size)
            prec1 = accuracy(output[i].data, target, topk=(1,))[0]
            top1_learner[i].update(prec1[0], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return [losses[i].avg for i in range(4)], [top1[i].avg for i in range(4)], [losses_learner[i].avg for i in range(num_model)], [top1_learner[i].avg for i in range(num_model)], batch_time

def validate_loss(val_loader, model, criterion):
    """Perform validation on the validation set"""

    # initialization
    num_model = len(model)
    batch_time = AverageMeter()
    losses = []
    top1 = []
    lossMat = ()
    for i in range(num_model):
        model[i].eval()
        losses.append(AverageMeter())
        top1.append(AverageMeter())

    # prediction of all batches
    end = time.time()
    for batch_index, (input, target) in enumerate(val_loader):

        # read batch data
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # prediction of all models per batch
        lossBatch = []
        for i in range(num_model):

            # compute loss
            modeli = model[i]
            _, outputi = modeli(input_var)
            lossi = criterion(outputi, target_var)
            lossBatch.append(lossi.data[0])

            # measure accuracy and record loss
            prec1 = accuracy(outputi.data, target, topk=(1,))[0]
            losses[i].update(lossi.data[0], input.size(0))
            top1[i].update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # concatenate loss of different models
        lossMat = lossMat + (np.asarray(lossBatch), )

    lossMat = np.vstack(lossMat).T
    lossAvg = [losses[i].avg for i in range(num_model)]
    PrecAvg = [top1[i].avg for i in range(num_model)]

    return lossMat, lossAvg, PrecAvg

    #     if i % args.print_freq == 0:
    #         print('Test: [{0}/{1}]\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
    #                   j, len(val_loader), batch_time=batch_time, loss=losses,
    #                   top1=top1))

    # print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # # log to TensorBoard
    # if args.tensorboard:
    #     log_value('val_loss', losses.avg, epoch)
    #     log_value('val_acc', top1.avg, epoch)

def validate_loss_old(train_set, model, criterion, epoch, cluster_nn):
    """Perform validation on the validation set"""
    losses = AverageMeter()
    #labels = np.unique(cluster_label)
    num_cluster = len(cluster_nn)
    loss_cluster = np.array([])
    kwargs = {'num_workers': 2, 'pin_memory': True}

    # switch to evaluate mode
    model.eval()

    # evaluate loss by cluster
    for ll in range(num_cluster):
        csampler = torch.utils.data.sampler.SubsetRandomSampler([cluster_nn[ll]])
        loss_loader = torch.utils.data.DataLoader(train_set,
            batch_size=1, sampler=csampler, **kwargs)
        for i, (input, target) in enumerate(loss_loader):
            target = target.cuda(async=True)
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            losses.update(loss.data[0], input.size(0))
        loss_cluster = np.append(loss_cluster, losses.avg)

    return loss_cluster

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 after 60, 120 and 160 epochs"""
    # lr = args.lr * ((0.5 ** int(epoch > 60)) * (0.5 ** int(epoch > 120))* (0.5 ** int(epoch > 160)))
    lr = args.lr * (0.92 ** int(epoch % 3 == 0))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def get_cluster(X, pca_dim, num_cluster, dataset_name, savefile = True, topk = 1):

    n = X.shape[0]
    center_nn = np.array([])
    centers_ = ()

    # dimension reduction
    if issparse(X):
        print 'TruncatedSVD of sparse X', (n, X.shape[1])
        svd = TruncatedSVD(n_components=pca_dim, algorithm='randomized', n_iter=15)
        X_pca = svd.fit_transform(X)
        print 'TruncatedSVD finished'
    elif n > 10000:
        print 'PCA of data size', n
        pca = PCA(n_components = pca_dim, svd_solver='randomized')
        X_pca = pca.fit_transform(X)
        print 'PCA finished'
    else:
        X_pca = X
        print 'PCA not applied'

    # clustering
    print 'k-means to', num_cluster, 'clusters'
    kmeans = MiniBatchKMeans(n_clusters = num_cluster, max_iter = 100, init_size = 3*num_cluster).fit(X_pca.astype('float64'))    
    labels_ = kmeans.labels_.astype('int32')
    labels_ = np.array([np.where(labels_ == i)[0].astype('int32') for i in range(num_cluster)])
    labels_weight = np.asarray(map(len, labels_))
    labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))
    nnz_ind = np.where(labels_weight != 0)[0]
    labels_ = labels_[nnz_ind]
    labels_weight = labels_weight[nnz_ind]
    
    for j in range(len(nnz_ind)):
        centers_ = centers_ + (np.mean(X[labels_[j], :], axis = 0),)
        center_nn = np.append(center_nn, labels_[j][np.argmin(euclidean_distances([kmeans.cluster_centers_[nnz_ind[j]]], X_pca[labels_[j]]))])
    centers_ = np.vstack(centers_)

    if savefile:
        np.savetxt(dataset_name + '_kmeans_labels.txt', cluster_label)
        np.savetxt(dataset_name + '_kmeans_centers.txt', cluster_centers)
        np.savetxt(dataset_name + '_center_nn.txt', center_nn)
        labels_, labels_weight, centers_, center_nn = [],[],[],[]
    else:
        return labels_, labels_weight, centers_, center_nn.astype('int32')

if __name__ == '__main__':
    main()