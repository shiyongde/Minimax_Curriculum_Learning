from __future__ import division
from numpy import *
from matplotlib.pyplot import *
from sklearn import *
import heapq
#from facloc import facloc
from facloc_graph import facloc_graph
#from satcoverage import satcoverage
from concavefeature import concavefeature
#from setcover import setcover 
#from submodular_coreset import submodular_coreset
#from videofeaturefunc import videofeaturefunc


class greedy:

	def __init__(self, X, func = 'facloc', func_parameter = ['euclidean_gaussian', 15], save_memory = [True, 8], offset = []):

		self.nn = []
		self.obj = 0
		if len(offset) == 0:
			self.offset = [0]*X.shape[0]
		else:
			self.offset = list(offset)

		if func == 'facloc':
			self.f = facloc_graph(X, func_parameter[0], func_parameter[1])
		elif func == 'satcoverage':
			self.f = satcoverage(X, func_parameter[0], func_parameter[1])
		elif func == 'concavefeature':
			self.f = concavefeature(X, func_parameter)
		elif func == 'setcover':
			self.f = setcover(X, func_parameter[0], func_parameter[1])
		elif func == 'videofeaturefunc':
			self.f = videofeaturefunc(X)
			X = X[1]
		else:
			print 'Function can only be facloc, satcoverage, concavefeature, setcover or videofeaturefunc'

		#print 'finish building submodular function for data size', X.shape[0], '\n'

		if save_memory[0] and size(X, 0) > 350:
			C = submodular_coreset(X, save_memory[1], self.f)
			self.V = C()
			print 'finish building submodular coreset of size', len(self.V), '\n' 
		else:
			self.V = range(X.shape[0])
		self.Vsize = len(self.V)

		# self.nn = f.evaluate(self.V) - asarray([f.evaluate(self.V.remove(x)) for x in self.V])
		# nobj = asarray([f.evaluate(x) for x in self.V])
		# self.V = self.V[nobj >= self.nn[K]]

		#print 'building heap...'
		self.S = []
		# print 'f(V) =', self.f.evaluate(self.V)[1]
		nn_obj = [subtract(0, self.f.evaluate([x])[1]+self.offset[x]) for x in self.V]
		# nn_obj = asarray(-map(f.evaluate, arange(X.shape[0]).reshape((X.shape[0], 1)).tolist())).reshape((1, X.shape[0])).tolist()
		self.heap_obj = zip(nn_obj, self.V)
		heapq.heapify(self.heap_obj)
		#print 'finish building heap'

		# self._update_set()

	def __call__(self, k, lazy = True):

		if k >= len(self.V):

			self.S = self.V
			self.obj = self.f.evaluate(self.S)[1] + sum(asarray(self.offset)[self.S])

		else:

			while (len(self.S) < k and len(self.V) > 0):

				self._update_set()

				if lazy:
					self._lazy_update_heap()
				else:
					self._update_heap()

				#print self.S

		#print 'Objective function = ', self.obj
		#print 'Solution Set = ', self.S		

		return self.S, self.obj, self.Vsize

	def _update_set(self):

		s = heapq.heappop(self.heap_obj)
		self.S.append(s[1])
		self.obj -= s[0]
		self.V.remove(s[1])
		if len(self.nn) == 0:
			self.nn = self.f.evaluate([s[1]])[0]
		else:
			self.nn = self.f.evaluate_incremental(self.nn, s[1])[0]

	def _update_heap(self):

		nn_obj = [subtract(self.obj, self.f.evaluate_incremental(self.nn, x)[1]+self.offset[x]) for x in self.V]
		self.heap_obj = zip(nn_obj, self.V)
		heapq.heapify(self.heap_obj)

	def _lazy_update_heap(self):

		i = 0
		while True:

			s = heapq.nsmallest(2, self.heap_obj)
			if len(s) < 1:
				break
			else:
				s[0] = (subtract(self.obj, self.f.evaluate_incremental(self.nn, s[0][1])[1]+self.offset[s[0][1]]), s[0][1])
				i += 1

			if len(s) < 2 or s[0][0] <= s[1][0]:
				_ = heapq.heappushpop(self.heap_obj, s[0])
				break
			else:
				_ = heapq.heappushpop(self.heap_obj, s[0])
				# self.V.remove(s[0][1])
		#print 'Number of evaluations =', i