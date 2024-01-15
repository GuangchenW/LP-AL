import math
import numpy as np

from .base_evaluator import BaseEvaluator

class NLP_Batch(BaseEvaluator):
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

		utilities = -utilities
		nlp_util = utilities

		# Resample?
		for i in range(min(n_points, len(subset_points))):
			min_id = np.argmax(nlp_util)
			batch.append({
				"next": subset_points[min_id],
				"mean": mean[min_id],
				"variance": variance[min_id],
				"utility": nlp_util[min_id]
				})

			hammer = self.calc_hammer(subset_points, batch)
			nlp_util = np.multiply(utilities, hammer)

		return np.array(batch)

	def calc_hammer(self, candidates, batch):
		hammer = []
		batch_pts = np.array([b["next"] for b in batch])
		for c in candidates:
			square_norm = ((batch_pts-c)**2).sum(axis=1)
			min_id = np.argmin(square_norm)
			min_dist = np.sqrt(square_norm[min_id])
			hammer.append(self.hammer_func(min_dist, batch[min_id]["mean"]))
		return np.array(hammer)

	def hammer_func(self, dist, offset):
		#dist = np.linalg.norm(center-candidate)
		z = dist-min(2,abs(offset))
		phi = 0.5*math.erfc(-z+1)
		return phi
