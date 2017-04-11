from __future__ import print_function
from __future__ import division

import sys
import os
import time
import string
import random

import numpy as np
import matplotlib.pyplot as plt

def load_news20_result(dataset, datadir='/home/tianyizhou/Dropbox/sensor_room/cLearn/result'):

    news20_random = ()
    news20_spl = ()
    news20_spl_info = ()
    news20_spld = ()
    news20_spld_info = ()
    news20_cLearn = ()
    news20_cLearn_info = ()
    news20_cLearnk = ()
    news20_cLearnk_info = ()
    news20_maxmax = ()
    news20_minmin = ()

    for file in os.listdir(datadir):
        if file.startswith(dataset+'random') and file.endswith('.txt'):
            news20_random = news20_random + (np.loadtxt(os.path.join(datadir, file)),)
        elif file.startswith(dataset+'spl_') and file.endswith('.txt'):
            news20_spl = news20_spl + (np.loadtxt(os.path.join(datadir, file)),)
            info = file.replace(dataset+'spl_','').replace('_result.txt','').split('_')
            news20_spl_info = news20_spl_info + ([int(info[0]), float(info[1])],)
        elif file.startswith(dataset+'spld_') and file.endswith('.txt'):
            news20_spld = news20_spld + (np.loadtxt(os.path.join(datadir, file)),)
            info = file.replace(dataset+'spld_','').replace('_result.txt','').split('_')
            news20_spld_info = news20_spld_info + ([int(info[0]), float(info[1])],)
        elif file.startswith(dataset+'cLearn_k') and file.endswith('.txt'):
            news20_cLearnk = news20_cLearnk + (np.loadtxt(os.path.join(datadir, file)),)
            info = file.replace(dataset+'cLearn_k+','').replace('_result.txt','')
            news20_cLearnk_info = news20_cLearnk_info + (int(info),)
        elif file.startswith(dataset+'cLearn_maxmax') and file.endswith('.txt'):
            news20_maxmax = news20_maxmax + (np.loadtxt(os.path.join(datadir, file)),)
        elif file.startswith(dataset+'cLearn_minmin') and file.endswith('.txt'):
            news20_minmin = news20_minmin + (np.loadtxt(os.path.join(datadir, file)),)             
        elif file.startswith(dataset+'cLearn') and file.endswith('.txt'):
            news20_cLearn = news20_cLearn + (np.loadtxt(os.path.join(datadir, file)),)
            info = file.replace(dataset+'cLearn_','').replace('_result.txt','').split('_')
            if len(info) == 1 and info[0] == 'result.txt':
                news20_cLearn_info = news20_cLearn_info + (0,)
            elif len(info) == 1 and info[0] == 'nosub':
                news20_cLearn_info = news20_cLearn_info + (1,)
            elif len(info) == 2:
                news20_cLearn_info = news20_cLearn_info + ([int(info[0]), int(info[1])],)
        else:
            pass

    return news20_random, news20_spl, news20_spl_info, news20_spld, news20_spld_info, news20_cLearn, news20_cLearn_info, news20_cLearnk, news20_cLearnk_info, news20_minmin, news20_maxmax

def data_clean(random, spl, spld, cLearn, cLearnk, minmin, maxmax, clip=0):

    for trial in random:
        clip = max([clip, trial[0, 0]])
        #print(trial[0, 0])
    for trial in spl:
        clip = max([clip, trial[0, 0]])
        #print(trial[0, 0])
    for trial in spld:
        clip = max([clip, trial[0, 0]])
        #print(trial[0, 0])
    for trial in cLearn:
        clip = max([clip, trial[0, 0]])
        #print(trial[0, 0])
    for trial in cLearnk:
        clip = max([clip, trial[0, 0]])
    for trial in maxmax:
        clip = max([clip, trial[0, 0]])
    for trial in minmin:
        clip = max([clip, trial[0, 0]])

    print('clip=', clip)

    randomC = ()
    for trial in random:
        randomC = randomC + (trial[trial[:,0]>=clip,:],)
    splC = ()
    for trial in spl:
        splC = splC + (trial[trial[:,0]>=clip,:],)
    spldC = ()
    for trial in spld:
        spldC = spldC + (trial[trial[:,0]>=clip,:],)
    cLearnC = ()
    for trial in cLearn:
        cLearnC = cLearnC + (trial[trial[:,0]>=clip,:],)
    cLearnkC = ()
    for trial in cLearnk:
        cLearnkC = cLearnkC + (trial[trial[:,0]>=clip,:],)
    minminC = ()
    for trial in minmin:
        minminC = minminC + (trial[trial[:,0]>=clip,:],)
    maxmaxC = ()
    for trial in maxmax:
        maxmaxC = maxmaxC + (trial[trial[:,0]>=clip,:],)

    return randomC, splC, spldC, cLearnC, cLearnkC, minminC, maxmaxC

def read_files(dataset, datadir = '/home/tianyizhou/Dropbox/sensor_room/cLearn/result', clip = 0):

    if dataset == 'news20':
        dataset = '20newsgroups_logistic_'
    elif dataset == 'mnist':
        dataset = 'mnist_lenet5_'
    elif dataset == 'cifar10':
        dataset = 'cifar10_convnet_'
    else:
        print('no dataset found!')
    random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info, minmin, maxmax = load_news20_result(dataset, datadir)
    random, spl, spld, cLearn, cLearnk, minmin, maxmax = data_clean(random, spl, spld, cLearn, cLearnk, minmin, maxmax, clip)

    return random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info, minmin, maxmax

def plt_error_vs_ntrain(dataset, error, ntrain, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info):

    if ntrain:
        xaxis = 0
        xlabel = 'Number of used training samples'
        fname1 = 'ntrain'
    else:
        xaxis = 1
        xlabel = 'Number of gradient computation on one sample (sample-wise passes)'
        fname1 = 'npass'

    if error:
        fname0 = 'err'
        ylabel = 'Error rate (%)'
    else:
        fname0 = 'loss'
        ylabel = 'Loss'


    if dataset == 'news20' and error:
        ttid = [4,5]
    elif dataset == 'mnist' and error:
        ttid = [2,4]
    elif dataset == 'cifar10' and not error:
        ttid = [2,3]
    elif dataset == 'news20' and not error:
        ttid = [2,3]
    else:
        print('dataset or result not found!')

    ax = plt.figure(figsize = (18, 10))
    #ax = plt.figure(figsize = (18, 8))
    #ax = plt.figure(figsize = (13, 8))
    ax = plt.subplot(111)

    for i in range(len(random)):
        trial = random[i]
        # train err
        ax.plot(trial[:, xaxis], trial[:, ttid[0]], c = 'grey', ls = '-', marker = '.', linewidth=1, alpha=0.5)
        # test err
        ax.plot(trial[:, xaxis], trial[:, ttid[1]], c = 'grey', ls = '--', marker = '.', linewidth=1, alpha=0.5)

    spl_color = ['gold','darkorange','orange','goldenrod']
    for i in range(len(spl)):
        trial = spl[i]
        info = spl_info[i]
        # train err
        ax.plot(trial[:, xaxis], trial[:, ttid[0]], c = spl_color[i], ls = '-', marker = 's', markersize = 7, linewidth=2, alpha=0.65, label = 'SPL train:'+str(info[0])+','+str(info[1]))
        # test err
        ax.plot(trial[:, xaxis], trial[:, ttid[1]], c = spl_color[i], ls = '--', marker = 's', markersize = 7, linewidth=2, alpha=0.65, label = 'SPL test:'+str(info[0])+','+str(info[1]))

    spld_color = ['lime','cyan','sage','darkturquoise','darkgreen', 'deepskyblue']        
    for i in range(len(spld)):
        trial = spld[i]
        info = spld_info[i]
        # train err
        ax.plot(trial[:, xaxis], trial[:, ttid[0]], c = spld_color[i], ls = '-', marker = '^', markersize = 7, linewidth=2, alpha=0.65, label = 'SPLD train:'+str(info[0])+','+str(info[1]))
        # test err
        ax.plot(trial[:, xaxis], trial[:, ttid[1]], c = spld_color[i], ls = '--', marker = '^', markersize = 7, linewidth=2, alpha=0.65, label = 'SPLD test:'+str(info[0])+','+str(info[1]))

    cLearn_color = ['lightcoral','maroon','fuchsia','crimson','deeppink', 'darkviolet']
    j = 0        
    for i in range(len(cLearn)):
        trial = cLearn[i]
        info = cLearn_info[i]
        if cLearn_info[i] == 0:
            cc = 'red'
            ll = ['MCL ', 'w/o random']
            mk = 'o'
            #trial[:, ttid[1]] -= 1.5
        elif cLearn_info[i] == 1:
            cc = 'blue'
            ll = ['MCL ', 'w/o submodular']
            mk = 'o'
            #trial[:, ttid[0]] += 0.1
            #trial[:, ttid[1]] += 0.1
        else:
            cc = cLearn_color[j]
            ll = ['MCL+random ', str(info[0])+','+str(info[1])]
            mk = 'v'
            #trial[:, ttid[1]] -= 1.5
            j += 1
        # train err
        ax.plot(trial[:, xaxis], trial[:, ttid[0]], c = cc, ls = '-', marker = mk, markersize = 7, linewidth=2, alpha=0.65, label = ll[0] + 'train:'+ll[1])
        # test err
        ax.plot(trial[:, xaxis], trial[:, ttid[1]], c = cc, ls = '--', marker = mk, markersize = 7, linewidth=2, alpha=0.65, label = ll[0] + 'test:'+ll[1])

    cLearnk_color = ['royalblue','steelblue','navy','dodgerblue','lightskyblue']
    for i in range(len(cLearnk)):
        trial = cLearnk[i]
        info = cLearnk_info[i]
        # train err
        ax.plot(trial[:, xaxis], trial[:, ttid[0]], c = cLearnk_color[i], ls = '-', marker = 'd', markersize = 7, linewidth=2, alpha=0.65, label = 'MCL+k train:'+str(info))
        # test err
        ax.plot(trial[:, xaxis], trial[:, ttid[1]], c = cLearnk_color[i], ls = '--', marker = 'd', markersize = 7, linewidth=2, alpha=0.65, label ='MCL+k test:'+str(info))


    plt.grid()
    plt.legend(fontsize='x-small', loc = 1)
    plt.ylabel(ylabel)   
    plt.xlabel(xlabel)
    #plt.xlim([5050, 50200])
    plt.xlim([0, 1e+7])
    plt.yscale('log')
    plt.ylim([0.1, 3])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('./result/'+dataset+'_'+fname0+'_vs_'+fname1+'.eps', format = 'eps', bbox_inches='tight')
    plt.show()

def plt_error_cifar10(dataset, ntrain, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info, minmin, maxmax):

    if ntrain:
        xaxis = 0
        xlabel = 'Number of used training samples'
        fname1 = 'ntrain'
    else:
        xaxis = 1
        xlabel = 'Number of gradient computation on one sample (sample-wise passes)'
        fname1 = 'npass'

    #ax = plt.figure(figsize = (18, 8))
    ax = plt.figure(figsize = (13, 8))
    ax = plt.subplot(111)

    for i in range(len(random)):
        trial = random[i]
        # test err
        ax.plot(trial[:, xaxis], 100-trial[:, -1], c = 'grey', ls = '--', marker = '.', linewidth=1, alpha=0.5)

    spl_color = ['gold','darkorange','orange','goldenrod']
    for i in range(len(spl)):
        trial = spl[i]
        info = spl_info[i]
        # test err
        ax.plot(trial[:, xaxis], 100-trial[:, -1], c = spl_color[i], ls = '--', marker = 's', markersize = 7, linewidth=2, alpha=0.65, label = 'SPL test:'+str(info[0])+','+str(info[1]))

    spld_color = ['lime','cyan','sage','darkturquoise','darkgreen', 'deepskyblue']        
    for i in range(len(spld)):
        trial = spld[i]
        info = spld_info[i]
        # test err
        ax.plot(trial[:, xaxis], 100-trial[:, -1], c = spld_color[i], ls = '--', marker = '^', markersize = 7, linewidth=2, alpha=0.65, label = 'SPLD test:'+str(info[0])+','+str(info[1]))

    cLearn_color = ['lightcoral','maroon','fuchsia','crimson','deeppink', 'darkviolet']
    j = 0        
    for i in range(len(cLearn)):
        trial = cLearn[i]
        info = cLearn_info[i]
        if cLearn_info[i] == 0:
            cc = 'red'
            ll = ['MCL ', 'w/o random']
            mk = 'o'
            #trial[:, ttid[1]] -= 1.5
        elif cLearn_info[i] == 1:
            cc = 'blue'
            ll = ['MCL ', 'w/o submodular']
            mk = 'o'
            #trial[:, ttid[0]] += 0.5
            #trial[:, ttid[1]] += 0.5
        else:
            cc = cLearn_color[j]
            ll = ['MCL+random ', str(info[0])+','+str(info[1])]
            mk = 'v'
            #trial[:, ttid[1]] -= 1.5
            j += 1
        # test err
        ax.plot(trial[:, xaxis], 100-trial[:, -1], c = cc, ls = '--', marker = mk, markersize = 7, linewidth=2, alpha=0.65, label = ll[0] + 'test:'+ll[1])

    cLearnk_color = ['royalblue','steelblue','navy','dodgerblue','lightskyblue']
    for i in range(len(cLearnk)):
        trial = cLearnk[i]
        info = cLearnk_info[i]
        # test err
        ax.plot(trial[:, xaxis], 100-trial[:, -1], c = cLearnk_color[i], ls = '--', marker = 'd', markersize = 7, linewidth=2, alpha=0.65, label ='MCL+k test:'+str(info))

    minmin_color = ['saddlebrown','steelblue','navy','dodgerblue','lightskyblue']
    for i in range(len(minmin)):
        trial = minmin[i]
        # test err
        ax.plot(trial[:, xaxis], 100-trial[:, -1], c = minmin_color[i], ls = '--', marker = '+', markersize = 7, linewidth=2, alpha=0.65, label ='minmin+k test:8')

    maxmax_color = ['olivedrab','steelblue','navy','dodgerblue','lightskyblue']
    for i in range(len(maxmax)):
        trial = maxmax[i]
        # test err
        ax.plot(trial[:, xaxis], 100-trial[:, -1], c = maxmax_color[i], ls = '--', marker = 'x', markersize = 7, linewidth=2, alpha=0.65, label ='maxmax+k test:8')


    plt.grid()
    plt.legend(fontsize='medium', loc = 1)
    plt.ylabel('Error rate (%)')   
    plt.xlabel(xlabel)
    plt.xlim([5600, 50200])
    #plt.xlim([0, 7e+6])
    #plt.yscale('log')
    #plt.ylim([0, 3])

    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('./result1/'+dataset+'_err_vs_'+fname1+'.eps', format = 'eps', bbox_inches='tight')
    plt.show()    

if __name__ == '__main__':

    #random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info = read_files('news20', '/Users/tianyi/Dropbox/sensor_room/cLearn/result')
    #plt_error_vs_ntrain('news20', True, True, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)
    #plt_error_vs_ntrain('news20', False, True, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)
    #plt_error_vs_ntrain('news20', True, False, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)
    #plt_error_vs_ntrain('news20', False, False, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)

    #random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info = read_files('mnist', '/Users/tianyi/Dropbox/sensor_room/cLearn/result')
    #plt_error_vs_ntrain('mnist', True, True, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)
    #plt_error_vs_ntrain('mnist', True, False, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)

    random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info, minmin, maxmax = read_files('cifar10','/Users/tianyi/Dropbox/sensor_room/cLearn/result1')
    #plt_error_vs_ntrain('cifar10', False, True, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)
    #plt_error_vs_ntrain('cifar10', False, False, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info)
    plt_error_cifar10('cifar10', True, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info, minmin, maxmax)
    #plt_error_cifar10('cifar10', False, random, spl, spl_info, spld, spld_info, cLearn, cLearn_info, cLearnk, cLearnk_info, minmin, maxmax)

# from plot_cLearn import *