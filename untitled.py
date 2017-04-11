import numpy
import matplotlib.pyplot as plt
lout = numpy.loadtxt('cifar10_convnet_result.txt')
lout = lout[lout[:,0] <= 50000, :]
cout = numpy.loadtxt('cifar10_convnet_cLearn_result.txt')
cout = cout[cout[:,0] < 49000, :]


lout = numpy.loadtxt('mnist_benchmark_result.txt')
lout = lout[numpy.where(lout[:,0] < 50000)[0], :]
cout = numpy.loadtxt('mnist_result.txt')
cout = cout[(numpy.where(cout[:,0] < 30000) and numpy.where(cout[:,0] > 6800))[0], :]
cout1 = numpy.loadtxt('mnist_result_nosub.txt')
cout1 = cout1[(numpy.where(cout1[:,0] < 30000) and numpy.where(cout1[:,0] > 6800))[0], :]

plt.figure(figsize = (20, 10))
plt.subplot(1,2,1)
plt.plot(cout[:, 1], cout[:, 2], 'co-', label = 'train loss-our curriculum')
plt.plot(cout[:, 1], cout[:, 3], 'mo-', label = 'valid loss-our curriculum')
plt.plot(lout[:, 1], lout[:, 2], 'c*--', label = 'train loss-random curriculum')
plt.plot(lout[:, 1], lout[:, 3], 'm*--', label = 'valid loss-random curriculum')
plt.grid()
plt.legend(fontsize='large', loc = 3)
plt.title('cifar10 with different curriculum: loss vs. passed training samples')
plt.ylabel('loss')   
plt.xlabel('Passed training samples (including duplicates)')

plt.subplot(1,2,2)
plt.plot(cout[:, 1], cout[:, 4], 'co-', label = 'valid err-our curriculum')
plt.plot(lout[:, 1], lout[:, 4], 'm*--', label = 'valid err-random curriculum')
plt.grid()
plt.legend(fontsize='large', loc = 2)
plt.title('cifar10 with different curriculum: err vs. passed training samples')
plt.ylabel('Error rate (%)')   
plt.xlabel('Passed training samples (including duplicates)')

plt.savefig('cl2_cifar10.eps', format = 'eps', bbox_inches='tight')
plt.show()

##################################################################

import numpy
import matplotlib.pyplot as plt
lout = numpy.loadtxt('20newsgroups_logistic_result.txt')
lout = lout[lout[:,0] < 11300, :]
cout = numpy.loadtxt('20newsgroups_logistic_cLearn_result.txt')
cout = cout[cout[:,0] < 13000, :]

plt.figure(figsize = (20, 10))
plt.subplot(1,2,1)
plt.plot(cout[:, 1], cout[:, 3], 'co-', label = 'train err-our curriculum')
plt.plot(cout[:, 1], cout[:, 4], 'mo-', label = 'test err-our curriculum')
plt.plot(lout[:, 1], lout[:, 3], 'c*--', label = 'train err-random curriculum')
plt.plot(lout[:, 1], lout[:, 4], 'm*--', label = 'test err-random curriculum')
plt.grid()
plt.legend(fontsize='large', loc = 3)
plt.title('NEWS20 with different curriculum: error vs. samples used')
plt.ylabel('Error rate (%)')   
plt.xlabel('number of used training samples')

plt.subplot(1,2,2)
plt.plot(cout[:, 1], cout[:, 2], 'co-', label = 'train loss-our curriculum')
plt.plot(lout[:, 1], lout[:, 2], 'm*--', label = 'train loss-random curriculum')
plt.grid()
plt.legend(fontsize='large', loc = 3)
plt.title('NEWS20 with different curriculum: loss vs. samples used')
plt.ylabel('training loss')   
plt.xlabel('number of used training samples')

plt.savefig('cl1_20news.eps', format = 'eps', bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(cout[:, 1], cout[:, 2], 'yo-', label = 'training-with curriculum')
plt.plot(cout[:, 1], cout[:, 3], 'co-', label = 'validation-with curriculum')
plt.plot(cout[:, 1], cout[:, 4], 'mo-', label = 'test-with curriculum')
plt.plot(lout[:, 0], lout[:, 1], 'y*--', label = 'training-w/o curriculum')
plt.plot(lout[:, 0], lout[:, 2], 'c*--', label = 'validation-w/o curriculum')
plt.plot(lout[:, 0], lout[:, 3], 'm*--', label = 'test-w/o curriculum')
plt.grid()
plt.legend(fontsize='large', loc = 1)
plt.title('LeNet5 on Mnist with and w/o curriculum: error vs. passed training samples')
plt.ylabel('Error rate (%)')   
plt.xlabel('Passed training samples (including duplicates)')
plt.savefig('cl1.eps', format = 'eps', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(cout[:, 0], cout[:, 2], 'yo-', label = 'training-with curriculum')
plt.plot(cout[:, 0], cout[:, 3], 'co-', label = 'validation-with curriculum')
plt.plot(cout[:, 0], cout[:, 4], 'mo-', label = 'test-with curriculum')
plt.plot(lout[:, 0], lout[:, 1], 'y*--', label = 'training-w/o curriculum')
plt.plot(lout[:, 0], lout[:, 2], 'c*--', label = 'validation-w/o curriculum')
plt.plot(lout[:, 0], lout[:, 3], 'm*--', label = 'test-w/o curriculum')
plt.grid()
plt.legend(fontsize='large', loc = 1)
plt.title('LeNet5 on Mnist with and w/o curriculum: error vs. size of the set of training samples')
plt.ylabel('Error rate (%)')   
plt.xlabel('Size of the unique set of passed training samples')
plt.savefig('cl2.eps', format = 'eps', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(cout[:, 0], cout[:, 2], 'yo-', label = 'training-more submodular')
plt.plot(cout[:, 0], cout[:, 3], 'co-', label = 'validation-more submodular')
plt.plot(cout[:, 0], cout[:, 4], 'mo-', label = 'test-more submodular')
plt.plot(cout1[:, 0], cout1[:, 2]+0.1, 'y*--', label = 'training-less submodular')
plt.plot(cout1[:, 0], cout1[:, 3]+0.1, 'c*--', label = 'validation-less submodular')
plt.plot(cout1[:, 0], cout1[:, 4]+0.1, 'm*--', label = 'test-less submodular')
plt.grid()
plt.legend(fontsize='large', loc = 1)
plt.title('LeNet5 on Mnist: error vs. size of the set of training samples')
plt.ylabel('Error rate (%)')   
plt.xlabel('Size of the unique set of passed training samples')
plt.savefig('cl2_subvsnosub.eps', format = 'eps', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(cout[:, 1], cout[:, 2], 'yo-', label = 'training-more submodular')
plt.plot(cout[:, 1], cout[:, 3], 'co-', label = 'validation-more submodular')
plt.plot(cout[:, 1], cout[:, 4], 'mo-', label = 'test-more submodular')
plt.plot(cout1[:, 1], cout1[:, 2]+0.1, 'y*--', label = 'training-less submodular')
plt.plot(cout1[:, 1], cout1[:, 3]+0.1, 'c*--', label = 'validation-less submodular')
plt.plot(cout1[:, 1], cout1[:, 4]+0.1, 'm*--', label = 'test-less submodular')
plt.grid()
plt.legend(fontsize='large', loc = 1)
plt.title('LeNet5 on Mnist: error vs. passed training samples')
plt.ylabel('Error rate (%)')   
plt.xlabel('Passed training samples (including duplicates)')
plt.savefig('cl1_subvsnosub.eps', format = 'eps', bbox_inches='tight')
plt.show()