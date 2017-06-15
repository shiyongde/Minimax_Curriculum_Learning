import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
#from scipy.spatial.distance import cdist
from scipy.sparse import issparse
#from sklearn.manifold import TSNE

import wideresnet as wrn
from models import *

#from facloc_graph import facloc_graph
#from satcoverage import satcoverage
#from concavefeature import concavefeature
#from setcover import setcover
from submdl2D import submdl_teach_welfare
from greedy2D import greedy2D
#from randomGreedy import randomGreedy

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run for learner')
parser.add_argument('--epochs4loss', default=20, type=int,
                    help='number of total epochs to run for loss predictor')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=1, type=int,
                    help='save frequency (default: 1)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.5, type=float,
                    help='dropout probability (default: 0.5)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--num_cluster', default=600, type=int,
                    help='Number of clusters (default: 1000)')
parser.add_argument('--num_learner_per_cluster', default=5, type=int,
                    help='Initial number of learners assigned to each sample (default: 5)')
parser.add_argument('--deltak', default=8, type=int,
                    help='Increased number of clusters in training set per epoch (default: 6)')
parser.add_argument('--loss_weight', default=1.5e+9, type=float,
                    help='Initial weight of loss term (default: 1.5e+9)')
parser.add_argument('--curriculum_rate', default=0.03, type=float,
                    help='Increasing ratio of loss weight (default: 0.03)')
parser.add_argument('--epoch_iters', default=30, type=int,
                    help='Number of iterations per epoch (default: 40)')
parser.add_argument('--stain_factor', default=60.0, type=float,
                    help='Stain factor of learned clusters (default: 60.0)')
parser.add_argument('--num_learner', default=10, type=int,
                    help='number of learners/models (default: 10)')
parser.add_argument('--func', default='concavefeature', type=str,
                    help='Submodular function for diversity regularization (default: concavefeature)')
parser.add_argument('--func_parameter', default=0.5, type=float,
                    help='Parameter of submodular function (default: 0.5)')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    # data preprocessing
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    # train/test set
    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_set = datasets.__dict__[args.dataset.upper()]('~/Downloads/data', train=True, download=True,
        transform=transform_train)
    train_set_copy = datasets.__dict__[args.dataset.upper()]('~/Downloads/data', train=True, download=True,
        transform=transform_train)
    train_label = train_set.train_labels
    # train_lossMat = np.zeros(args.num_learner, args.num_cluster)
    n_train = len(train_label)
    train_label = np.copy(train_label)
    fea_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args.batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('~/Downloads/data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # learner models
    model = wrn.WideResNet(args.layers, args.num_learner, args.widen_factor, dropRate=args.droprate)
    for i in range(args.num_learner):
        model.append(wrn.WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100, args.widen_factor, dropRate=args.droprate))

    for i in range(args.num_learner):
        # get the number of model parameters
        print('Model['+str(i)+']: number of parameters: {}'.format(
            sum([p.data.nelement() for p in model[i].parameters()])))
        model[i] = model[i].cuda()

    # loss predictor model
    net = VGG('VGG11')
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    net.cuda()

    # for training on multiple GPUs.
    # CUDA_VISIBLE_DEVICES = 0,1,2,3
    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # loss function for learner model
    criterion = nn.CrossEntropyLoss().cuda()
    # loss function for loss predictor
    net_loss = nn.MSELoss().cuda()

    # optimizer for learner models
    optimizer = []
    for i in range(args.num_learner):
        optimizer.append(torch.optim.SGD(model[i].parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay))
    # optimizer for loss predictor
    optimizer_net = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # initialization
    passed_index = [np.array([]).astype('int32')] * args.num_learner
    passes = np.zeros(args.num_learner)
    output_seq = ()
    submodular_time = 0
    training_time = 0

    # clustering
    labels_, cluster_centers_, center_nn = dataGroup0(train_fea, 0, args.num_cluster, args.dataset, savefile = False)
    labels_ = [np.where(labels_ == i)[0].astype('int32') for i in range(args.num_cluster)]
    center_nn = center_nn.astype('int32')       
    labels_weight = np.array([len(np.where(labels_==i)[0]) for i in np.unique(labels_)])
    labels_weight = np.divide(labels_weight,float(np.max(labels_weight)))

    # Initialize submodular function and greedy algorithm
    rewardMat = np.zeros(args.num_learner, args.num_cluster)
    SubmodularF = submdl_teach_welfare(cluster_centers, rewardMat)
    greedyAlg = greedy2D(SubmodularF)
    topk_F = args.num_learner * sum(np.partition(greedyAlg.sinGain, -args.num_learner_per_cluster)[-args.num_learner_per_cluster:])

    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # begin to solve minimax problem
        epoch_end = time.time()

        for iters in range(args.epoch_iters):

            # generate training set ---------------------------------------------------------------------------------
            if epoch % 10 == 0 and epoch >= 200:
                # random
                train_subset = []
                for i in range(args.num_learner):
                    train_subset.append(np.random.choice(n_train, args.batch_size*3, replace=False).tolist())
                    passed_index[i] = np.unique(np.append(passed_index[i], train_subset[i]))

            elif iters % 10 == 0:               
                # submodular teacher welfare
                submodular_start_time = time.time()

                # compute loss of cluster centroids and transform to reward
                csampler = torch.utils.data.sampler.SubsetRandomSampler(cluster_nn)
                loss_loader = torch.utils.data.DataLoader(train_set, batch_size=1, sampler=csampler, **kwargs)
                lossMat, lossAvg, PrecAvg = validate_loss(loss_loader, model, criterion)
                rewardMat = (lossMat.max(axis = 0) - lossMat) * (labels_weights * args.loss_weight)

                # submodular optimization to select clusters for each learner
                train_clusters, greedyObj, Vsize = greedyAlg(args.num_learner_per_cluster, lossMat)
                # for debug use
                topk_L = sum([sum(np.partition(rewardMat[:, i], -args.num_learner_per_cluster)[-args.num_learner_per_cluster:]) for i in range(n_train)])
                print 'topk of L and F:', topk_L, topk_F

                # transform from selected clusters to training sample index
                train_subset = []
                for i in range(args.num_learner):
                    train_subset.append(np.concatenate([labels_[j] for j in train_clusters[i]]).tolist())
                    passed_index[i] = np.unique(np.append(passed_index[i], train_subset[i]))

                submodular_time += (time.time() - submodular_start_time)
            # generate training set ---------------------------------------------------------------------------------

            # training stage ----------------------------------------------------------------------------------------
            training_start_time = time.time()
            loss_iter = np.zeros(args.num_learner)
            for i in range(args.num_learner):

                # change learning rate
                print('Now update model'+str(i))
                adjust_learning_rate(optimizer[i], epoch)

                # load training set for each model
                csampler = torch.utils.data.sampler.SubsetRandomSampler(train_subset[i])
                train_loader = torch.utils.data.DataLoader(train_set,
                    batch_size=args.batch_size, sampler=csampler, **kwargs)

                # train learner i
                loss_iter[i] = train(train_loader, model[i], criterion, optimizer[i], epoch)

                # record passes
                passes[i] += len(train_subset[i])

            training_time += (time.time() - training_start_time)
            print 'passes, unique_passes', passes, [len(passed_index[i]) for i in range(agrs.num_learner)]
            # training stage ----------------------------------------------------------------------------------------

        # change learning pace
        args.loss_weight *= args.curriculum_rate + 1
        args.num_learner_per_cluster = max([args.num_learner_per_cluster - args.deltak, 1])

        # save intermediate results
        if epoch % args.save_freq == 0:

            # train loss predictor ----------------------------------------------------------------------------------
            train_loss_loader = train_loss_loader(fea_loader, train_set_copy, model, criterion)
            regression_loss = np.zeros(args.epochs4loss)
            for epoch in range(args.epochs4loss):
                regression_loss[i] = train(train_loader, net, net_loss, optimizer_net, epoch)




            # evaluate on validation set
            _, prec1, loss1 = validate_ensemble(val_loader, model, criterion, epoch)
            # evaluate on training set
            _, prec2, loss2 = validate_ensemble(fea_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

            start = time.time()
            epoch_time = start - epoch_end
            total_time = start - end
            output_seq = output_seq + (np.array([len(passed_index),passes,loss2,loss1,(1-prec2) * 100.,(1-prec1) * 100.]),)

            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {epoch_time:.3f} ({total_time:.3f})\t'
                    'Loss {loss_train:.4f} ({loss_test:.4f})\t'
                    'Prec@1 {prec_train:.3f} ({prec_test:.3f})'.format(
                        epoch, len(passed_index), passes, 
                        epoch_time=epoch_time, total_time=total_time, 
                        loss_test=loss1, loss_train=loss2,
                        prec_test=prec1, prec_train=prec2))
    
    # save result to file
    output_seq = np.vstack(output_seq)
    np.savetxt('cifar100_wideresnet_ECL_result_3.txt', output_seq)

    # show final result
    print 'Best accuracy: ', best_prec1
    print 'SubmodularMax time: ', submodular_time
    print 'Training time: ', training_time
    print 'Total time: ', time.time() - end

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    # batch_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()

    # end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # print(loss.data[0])

        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    top1 = AverageMeter()

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
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg

def train_loss_loader(train_loader, train_set, model, criterion):

    # For learner models: switch to evaluate mode
    num_model = len(model)
    for i in range(num_model):
        model[i].eval()

    # generate regression target for loss predictor
    train_loss = ()
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output and confidence
        alloss = ()
        for i in range(num_model):
            modeli = model[i]
            _, outputi = modeli(input_var)
            alloss = alloss + (criterion(outputi, target_var), )
        train_loss = train_loss + (torch.stack(alloss, 0), )

    # preprocess of losses as target (clamp, -log)    
    train_loss = torch.stack(train_loss, 1)
    train_loss = -train_loss.clamp(1e-13, 4.0).log()
    train_set.train_labels = train_loss
    train_loss_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loss_loader

def validate_ensemble(val_loader, model, criterion, test=True):
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
    for i in range(num_model):
        model[i].eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output and confidence
        output = ()
        confid = ()
        alloss = ()
        for i in range(num_model):
            modeli = model[i]
            _, outputi = modeli(input_var)
            output = output + (outputi, )
            confidi = confidence(outputi)
            confid = confid + (confidi, )
            alloss = alloss + (nllloss(confidi, target_var), )

        # num_model x batch_size x num_class
        output = torch.stack(output, 0)
        confid = torch.stack(confid, 0)
        batch_size = confid.size(1)
        num_class = confid.size(2)
        # num_model x batch_size
        cinfid_max = torch.max(confid, 2).view(num_model, batch_size)
        alloss = torch.stack(alloss, 0)

        # measure accuracy and record loss of different ensemble methods

        # topk confidence (greedy)
        val_confid, ind_confid = torch.topk(confid_max, args.num_learner_per_cluster, 0)
        output_ensemble = output.gather(0, ind_confid.repeat(num_class, 1, 1).permute(1,2,0)).mean(0).squeeze(0)
        loss = criterion(output_ensemble, target_var)
        losses[0].update(loss.data[0], batch_size)
        prec1 = accuracy(output_ensemble.data, target, topk=(1,))[0]
        top1[0].update(prec1[0], batch_size)

        # topk smallest loss (oracle)
        val_loss, ind_loss = torch.topk(alloss, args.num_learner_per_cluster, 0, largest = False)
        output_ensemble = output.gather(0, ind_loss.repeat(num_class, 1, 1).permute(1,2,0)).mean(0).squeeze(0)
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
        val_ours, ind_ours = torch.topk(predloss, args.num_learner_per_cluster, 0, largest = False)
        output_ensemble = output.gather(0, ind_ours.repeat(num_class, 1, 1).permute(1,2,0)).mean(0).squeeze(0)
        loss = criterion(output_ensemble, target_var)
        losses[3].update(loss.data[0], batch_size)
        prec1 = accuracy(output_ensemble.data, target, topk=(1,))[0]
        top1[3].update(prec1[0], batch_size)        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0 and test:
           # print(set_name+': [{0}/{1}]\t'
                 # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                     # i, len(val_loader), batch_time=batch_time, loss=losses,
                     # top1=top1))

    #print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # if args.tensorboard:
    #     if test:
    #         log_value('val_loss', losses.avg, epoch)
    #         log_value('val_acc', top1.avg, epoch)
    #     else:
    #         log_value('train_loss', losses.avg, epoch)
    #         log_value('train_acc', top1.avg, epoch)           

    # if not test:
    #     print len(output_feature)
    #     output_feature = torch.cat(output_feature, 0)
    #     output_feature = output_feature.data.cpu().numpy()

    return [losses[i].avg for i in range(4)], [top1[i].avg for i in range(4)], batch_time

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
    for j, (input, target) in enumerate(val_loader):

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
    kwargs = {'num_workers': 1, 'pin_memory': True}

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

def dataGroup0(X, tsne_dim, num_cluster, dataset_name, savefile = True, topk = 1):

    print 'clustering...'
    n = X.shape[0]
    center_nn = np.array([])
    cluster_centers = ()
    if tsne_dim <= 0 and not issparse(X) and n <= 10000:
        X_tsne = X
    elif issparse(X) or n > 10000:
        if tsne_dim == 0:
            tsne_dim = 48
        print 'TruncatedSVD of data size', (n, X.shape[1])
        svd = TruncatedSVD(n_components=tsne_dim, algorithm='randomized', n_iter=10, random_state=42)
        X_tsne = svd.fit_transform(X)
        print 'finish TruncatedSVD.'
    else:
        print 'PCA of data size', n
        pca = PCA(n_components = tsne_dim)
        X_tsne = pca.fit_transform(X)
        print 'finish PCA.'
    print 'k-means to', num_cluster, 'clusters'
    kmeans = KMeans(n_clusters = num_cluster, max_iter = 50).fit(X_tsne.astype('float64'))
    cluster_label = kmeans.labels_
    for j in range(num_cluster):
        jIndex = np.where(cluster_label==j)[0]
        centerj = np.mean(X[jIndex, :], axis = 0)
        cluster_centers = cluster_centers + (centerj,)
        center_nn = np.append(center_nn, jIndex[np.argmin(euclidean_distances([kmeans.cluster_centers_[j]], X_tsne[jIndex]))])

    cluster_centers = np.vstack(cluster_centers)

    if savefile:
        np.savetxt(dataset_name + '_kmeans_labels.txt', cluster_label)
        np.savetxt(dataset_name + '_kmeans_centers.txt', cluster_centers)
        np.savetxt(dataset_name + '_center_nn.txt', center_nn)
        cluster_label, cluster_centers, center_nn = [],[],[]
    else:
        return cluster_label, cluster_centers, center_nn 

if __name__ == '__main__':
    main()
