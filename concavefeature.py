from __future__ import division
# from __future__ import print_function
from numpy import *
from matplotlib.pyplot import *
#from sklearn import *
#import sklearn.metrics.pairwise as simi
#import scipy.spatial.distance as scd
import scipy.sparse as sps

class concavefeature:

	def __init__(self, X, p = 0.5, weighted = False):

		self.p = p
		self.num_sample = X.shape[0]
		self.X = X
		if sps.issparse(X):
			self.sparse = True
		else:
			self.sparse = False
		if not self.sparse:
			# normalize
			xmin = amin(X, axis = 0)
			id_neg = where(xmin < 0)[0]
			if len(id_neg) > 0:
				for i in id_neg:
					X[:, i] -= xmin[i]
			# remove zero features
			self.w = sum(X, axis = 0)
			id_zero = where(self.w == 0)
			if len(id_zero) > 0:
				X = delete(X, id_zero, 1)
				self.w = delete(self.w, id_zero)
		# weighted
		if weighted:
			self.w *= 0.2*len(self.w)/sum(self.w)
		else:
			self.w = 1.0

	def evaluate(self, A):

		if self.sparse:
			nn_obj = array(self.X[A, :].sum(0))[0]
		else:
			nn_obj = sum(self.X[A, :], 0)
		obj = sum(self.w * power(nn_obj, self.p))
		return nn_obj, obj

	def evaluate_incremental(self, nn_obj, a):

		nn_obj = nn_obj + self.X[a, :]
		if self.sparse:
			nn_obj = array(nn_obj)[0]
		obj = sum(self.w * power(nn_obj, self.p))
		return nn_obj, obj

	def evaluate_decremental(self, nn_obj, a, A = []):

		nn_obj = nn_obj - self.X[a, :]
		if self.sparse:
			nn_obj = array(nn_obj)[0]
		obj = sum(self.w * power(nn_obj, self.p))
		return nn_obj, obj

	def evaluateV(self):

		nn_obj, obj = self.evaluate(range(self.num_sample))
		return nn_obj, obj