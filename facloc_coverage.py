from __future__ import division
# from __future__ import print_function
from numpy import *
from matplotlib.pyplot import *
from sklearn import *
import sklearn.metrics.pairwise as simi
import scipy.spatial.distance as scd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import time

class facloc_coverage:

	@staticmethod
	def update_nn(sim, row, col, nn_ind, nn_obj, a):

		# nn_obj_a = zeros(sim.shape[0])
		#start = time.time()
		ix = where(col == a)[0]
		#print 'ix has size', len(ix)
		#end = time.time()
		#print 'update_nn step1 ', end-start, ' seconds'

		nn_obj_a = sim[ix]
		change_ind_ix = where(nn_obj_a > nn_obj[row[ix]])[0]
		change_ind = row[ix[change_ind_ix]]
		nn_obj[change_ind] = nn_obj_a[change_ind_ix]
		nn_ind[change_ind] = a

		return nn_ind, nn_obj

	def __init__(self, X):

		self.num_learner = X.shape[0]
		self.num_sample = X.shape[1]

		if dis_metric == 'cosine':
			kgraph = neighbors.NearestNeighbors(n_neighbors=K, algorithm='brute', metric='cosine').fit(X).kneighbors_graph(mode='distance').tocoo(copy=False)
			row = kgraph.row
			col = kgraph.col
			sim = 1 - kgraph.data
		elif dis_metric == 'euclidean_gaussian':
			print("computing KNN graph...")
			kgraph = neighbors.NearestNeighbors(n_neighbors=K, algorithm='kd_tree', metric='euclidean').fit(X).kneighbors_graph(mode='distance').tocoo(copy=False)
			row = kgraph.row
			col = kgraph.col
			sim = exp(-divide(power(kgraph.data, 2), mean(kgraph.data)**2))
			print("finish KNN graph")
		elif dis_metric == 'euclidean_inverse':
			kgraph = neighbors.NearestNeighbors(n_neighbors=K, algorithm='kd_tree', metric='euclidean').fit(X).kneighbors_graph(mode='distance').tocoo(copy=False)
			row = kgraph.row
			col = kgraph.col
			sim = 1/kgraph.data
		else:
			print("Metrics can only be cosine, euclidean_gaussian, or euclidean_inverse")

		self.row = row
		self.col = col
		self.sim = sim
		self.nnset = list(set(col))

	def evaluate_incremental(self, nn, a):

		nn_ind = copy(nn[0])
		nn_obj = copy(nn[1])

		if a in self.nnset:
			nn_ind, nn_obj = facloc_graph.update_nn(self.sim, self.row, self.col, nn_ind, nn_obj, a)

		obj = sum(nn_obj)
		nnn = [nn_ind, nn_obj]

		return nnn, obj

	def evaluate(self, A):

		nn = [-ones(self.num_sample), zeros(self.num_sample)]
		obj = 0

		for a in A:
			if a in self.nnset:
				nn, obj = self.evaluate_incremental(nn, a)

		return nn, obj

	def evaluate_decremental(self, nn, a, A = []):

		if len(A) == 0:
			A = self.nnset

		change_ind = where(nn[0]==a)[0]
		nnn = copy(nn)
		A = asarray(A)

		for c in change_ind:
			row_c = where(self.row == c)[0]
			col_c = where(in1d(self.col[row_c], A[A != a]))[0]
			if len(col_c) == 0:
				nnn[0][c] = -1
				nnn[1][c] = 0;
				continue
			ind_c = row_c[col_c[argmax(self.sim[row_c[col_c]])]]
			nnn[0][c] = self.col[ind_c]
			nnn[1][c] = self.sim[ind_c]

		obj = sum(nnn[1])

		return nnn, obj

	def evaluateV(self):

		nn = [-ones(self.num_sample), zeros(self.num_sample)]

		for c in range(self.num_sample):
			row_c = where(self.row == c)[0]
			ind_c = row_c[argmax(self.sim[row_c])]
			nn[0][c] = self.col[ind_c]
			nn[1][c] = self.sim[ind_c]

		obj = sum(nn[1])

		return nn, obj

