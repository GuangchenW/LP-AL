import sys
import math
import numpy as np

from .base_evaluator import BaseEvaluator

class LP_Batch(BaseEvaluator):
	def __init__(self, acq_func, logger=None):
		super().__init__(acq_func=acq_func, logger=logger)
		self.L = float("nan")

	def set_grad(self, grad):
		self.grad = grad

	def set_L(self, L):
		self.L = min(10,L)

	def obtain_batch(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
		n_points
	):
		batch = []
		utilities = self.acq_func.acquire(subset_points, mean, variance, doe_input, doe_response)
		utilities += 1e-10 # To prevent log0 undefined
		utilities = np.log(utilities)

		# TEST HACK
		grad_norm = np.linalg.norm(self.grad, axis=1)

		for i in range(min(n_points, len(subset_points))):
			max_id = np.argmax(utilities)
			batch.append({
				"next": subset_points[max_id],
				"mean": mean[max_id],
				"variance": variance[max_id],
				"utility": utilities[max_id]
				})

			#self.L = grad_norm[max_id]
			utilities = self.apply_hammer(subset_points, batch[-1], utilities)
			#doe_input = np.append(doe_input, [subset_points[max_id]], axis=0)
			#doe_response = np.append(doe_response, [mean[max_id]])
			#k=i+1
			#mean, variance = self.model.fantasize(doe_input[-k:], doe_response[-k:], subset_points)

			#m, v = self.model.fantasize([[1,1],[2,2]], [10,5], [[1,1],[2,2]])
			#print(m,v)

		return np.array(batch)

	def apply_hammer(self, candidates, batch, utilities):
		hammered_util = []
		for i in range(len(candidates)):
			hammer = self.log_hammer_func(candidates[i], batch["next"], batch["mean"])
			util = utilities[i] + hammer
			hammered_util.append(util)
		return np.array(hammered_util)

	def log_hammer_func(self, candidate, center, offset):
		dist = np.linalg.norm(center-candidate)
		#dist = np.dot(center, candidate)/(np.linalg.norm(center)*np.linalg.norm(candidate))
		z = self.L*dist-abs(offset)
		phi = 0.5*math.erfc(-z)
		phi = max(sys.float_info.min, phi) # Prevent cases where erfc evaluates to 0
		return np.log(phi)

