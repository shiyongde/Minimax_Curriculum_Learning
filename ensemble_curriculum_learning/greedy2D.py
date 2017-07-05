from __future__ import division
from numpy import *
#from matplotlib.pyplot import *
# from sklearn import *
import heapq
# import time
# from copy import deepcopy
# from submdl2D import submdl_teach_welfare
#from facloc import facloc
#from facloc_graph import facloc_graph
#from satcoverage import satcoverage
#from concavefeature import concavefeature
#from setcover import setcover 
#from submodular_coreset import submodular_coreset
#from videofeaturefunc import videofeaturefunc


class greedy2D:

	def __init__(self, submdl, prune = True):

		self.f = submdl
		self.num_sample = self.f.num_sample
		self.num_learner = self.f.num_learner
		# self.Vsize = self.num_sample * self.num_learner
		# self.V = asarray(unravel_index(range(self.Vsize), (self.num_learner, self.num_sample), 'C')).T.tolist()
		self.prune = prune

		if self.prune:
			self.minGain, self.sinGain = self.f.compute_min_sin_gain_F()

		#print 'finish building submodular function for data size', X.shape[0], '\n'

		# self.nn = f.evaluate(self.V) - asarray([f.evaluate(self.V.remove(x)) for x in self.V])
		# nobj = asarray([f.evaluate(x) for x in self.V])
		# self.V = self.V[nobj >= self.nn[K]]

		#print 'building heap...'
		# print 'f(V) =', self.f.evaluate(self.V)[1]
		# nn_obj = [subtract(0, self.f.evaluate(pair2set([x]))[-1]) for x in self.V]
		# nn_obj = asarray(-map(f.evaluate, arange(X.shape[0]).reshape((X.shape[0], 1)).tolist())).reshape((1, X.shape[0])).tolist()
		# self.heap_obj = zip(nn_obj, self.V)
		# heapq.heapify(self.heap_obj)
		#print 'finish building heap'

		# self._update_set()

	def update_V(self):

		# pruning of V
		if self.prune:
			minGain = self.f.rewardMat + self.minGain
			sinGain = self.f.rewardMat + self.sinGain
			self.V = ()
			self.Vsize = 0
			for i in range(self.num_sample):
				left_index = where(sinGain[:, i] >= min(partition(minGain[:, i], -self.k)[-self.k:]))[0]
				self.V = self.V + (array([left_index, repeat(i, len(left_index))]).T, )
				self.Vsize += len(left_index)
			self.V = vstack(self.V).tolist()
		else:
			self.Vsize = self.num_sample * self.num_learner
			self.V = asarray(unravel_index(range(self.Vsize), (self.num_learner, self.num_sample), 'C')).T.tolist()				

	def __call__(self, k, rewardMat, lazy = True):

		# print 'set all to empty'
		self.k = k
		self.nn = []
		self.Lobj = []
		self.Fobj = []
		self.blacklist = []
		self.active_learner = []
		self.obj = 0
		# self.heap_obj = []
		# self.nn_obj = []
		self.S = [[] for i in range(self.num_learner)]
		self.num_sample_learner = [0] * self.num_sample
		# print 'all are empty now'

		self.f.update_reward(rewardMat)
		self.update_V()

		if self.k >= self.num_learner:

			self.S = [range(self.num_sample) for i in range(self.num_learner)]
			self.obj = self.f.evaluate(self.S)[-1]

		else:

			# start = time.time()
			while (len(self.V) > 0):

				if lazy:
					s0 = self._lazy_update_heap()
				else:
					s0 = self._update_heap()

				if len(s0) >= 1:
					self._update_set(s0, lazy)
				else:
					break

				# print 'S->', self.S
			# print 'time=', time.time() - start

		#print 'Objective function = ', self.obj
		#print 'Solution Set = ', self.S		

		return self.S, self.obj, self.Vsize

	def pair2set(self, a):
		a = asarray(a).T
		A = [a[1, a[0]==i].tolist() for i in range(self.num_learner)]
		return A

	def add2S(self, a, lazy):
		self.S[a[0]].append(a[1])
		self.active_learner = a[0]
		self.num_sample_learner[a[1]] += 1
		if self.num_sample_learner[a[1]] >= self.k:
			V_array = asarray(self.V).T
			V_ind = where(V_array[1, :] != a[1])[0]
			# self.V = [v for v in self.V if v[1] != a[1]]
			self.V = V_array[:, V_ind].T.tolist()
			if not lazy:
				self.nn_obj = self.nn_obj[V_ind]
			self.blacklist.append(a[1])
		else:
			a_ind = self.V.index(a)
			del self.V[a_ind]
			if not lazy:
				self.nn_obj = delete(self.nn_obj, a_ind)

	def _update_set(self, s, lazy):

		S_old = [list(self.S[i]) for i in range(self.num_learner)]
		self.add2S(s[1], lazy)
		self.obj -= s[0]
		if len(self.nn) == 0:
			self.nn, self.Lobj, self.Fobj, obj = self.f.evaluate(self.pair2set([s[1]]))
		elif not lazy:
			# self.nn, self.Lobj, self.Fobj, obj = self.f.evaluate_incremental(self.nn, self.Lobj, self.Fobj, S_old, s[1])
			nn_s, Lobj_s, Fobj_s, _= self.f.evaluate_incremental_fast(self.nn[s[1][0]], self.Fobj[s[1][0]], S_old, s[1])
			self.nn[s[1][0]] = nn_s
			self.Lobj[s[1][0]] += Lobj_s
			self.Fobj[s[1][0]] = Fobj_s

	def _update_heap(self):

		if len(self.nn) == 0:
			self.nn_obj = asarray([subtract(self.obj, self.f.evaluate(self.pair2set([x]))[-1]) for x in self.V])
		else:
			V_array = asarray(self.V).T
			V_active_ind = where(V_array[0, :] == self.active_learner)[0]
			V_active = V_array[:, V_active_ind].T.tolist()
			self.nn_obj[V_active_ind] = asarray([self.f.evaluate_incremental_fast(self.nn[x[0]], self.Fobj[x[0]], self.S, x)[-1] for x in V_active])

		min_ind = argmin(self.nn_obj)
		s0 = (self.nn_obj[min_ind], self.V[min_ind])

		return s0

	def _lazy_update_heap(self):

		if len(self.nn) == 0:
			nn_obj = [subtract(self.obj, self.f.evaluate(self.pair2set([x]))[-1]) for x in self.V]
			self.heap_obj = zip(nn_obj, self.V)
			heapq.heapify(self.heap_obj) 
			s0 = heapq.heappop(self.heap_obj)
		else:
			i = 0
			s0 = []
			while len(self.V) > 0:

				i += 1

				while len(self.heap_obj) > 0:
					if self.heap_obj[0][1][1] in self.blacklist:
						_ = heapq.heappop(self.heap_obj)
					else:
						break
				if len(self.heap_obj) >= 1: #and self.heap_obj[0][1][0] == self.active_learner:
					s0 = heapq.heappop(self.heap_obj)
					# nn, Lobj, Fobj, obj = self.f.evaluate_incremental(self.nn, self.Lobj, self.Fobj, self.S, s0[1])
					# s0 = (subtract(self.obj, obj), s0[1])
					nn_s, Lobj_s, Fobj_s, obj_s = self.f.evaluate_incremental_fast(self.nn[s0[1][0]], self.Fobj[s0[1][0]], self.S, s0[1])
					s0 = (obj_s, s0[1])
				else:
					break

				while len(self.heap_obj) > 0:
					if self.heap_obj[0][1][1] in self.blacklist:
						_ = heapq.heappop(self.heap_obj)
					else:
						break
				if len(self.heap_obj) >= 1:
					s1 = self.heap_obj[0]
				else:
					s1 = []

				if len(s1) == 0 or s0[0] <= s1[0]:
					# self.nn = deepcopy(nn)
					# self.Lobj = list(Lobj)
					# self.Fobj = list(Fobj)
					self.nn[s0[1][0]] = nn_s
					self.Lobj[s0[1][0]] += Lobj_s
					self.Fobj[s0[1][0]] = Fobj_s
					break
				else:
					_ = heapq.heappush(self.heap_obj, s0)
					# self.V.remove(s[0][1])
			#print 'Number of evaluations =', i

		# print 's0', s0

		# print 'size of priority queue:', len(self.heap_obj)

		return s0
