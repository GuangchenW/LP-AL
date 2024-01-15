import math
import numpy as np

from .base_evaluator import BaseEvaluator

class LP_Batch(BaseEvaluator):
	def __init__(self, acq_func, logger=None):
		super().__init__(acq_func=acq_func, logger=logger)

	def obtain_batch(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
		n_points
	):
		print("Subset size: %d" % len(subset_points))

		batch = []
		utilities = self.acq_func.acquire(subset_points, mean, variance, doe_input, doe_response)

		# HACK
		utilities = np.log(-utilities)

		# Resample?
		for i in range(min(n_points, len(subset_points))):
			min_id = np.argmax(utilities)
			batch.append({
				"next": subset_points[min_id],
				"mean": mean[min_id],
				"variance": variance[min_id],
				"utility": utilities[min_id]
				})

			utilities = self.apply_hammer(subset_points, batch[-1], utilities)
			#doe_input = np.append(doe_input, [subset_points[min_id]], axis=0)
			#doe_response = np.append(doe_response, [mean[min_id]])
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
		return hammered_util

	def log_hammer_func(self, candidate, center, offset):
		dist = np.linalg.norm(center-candidate)
		z = 1/0.15*dist-abs(offset)
		phi = 0.5*math.erfc(-z)
		return np.log(phi)

