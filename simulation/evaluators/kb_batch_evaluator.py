import sys
import math
import numpy as np
from scipy.stats import norm

from .base_evaluator import BaseEvaluator
from simulation.models import OrdinaryKriging

class KB_Batch(BaseEvaluator):
	def __init__(self, acq_func, logger=None):
		super().__init__(acq_func=acq_func, logger=logger)
		self.fantasy_model = None
		self.name = "kb"

	def set_grad(self, grad):
		self.grad = grad

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
		k = min(n_points, len(subset_points))
		if not self.fantasy_model:
			ndim = subset_points.shape[1]
			self.fantasy_model = OrdinaryKriging(n_dim=ndim)

		for i in range(k):
			utilities = self.acq_func.acquire(subset_points, mean, variance, doe_input, doe_response)
			max_id = np.nanargmax(utilities)
			batch.append({
				"next": subset_points[max_id],
				"mean": mean[max_id],
				"variance": variance[max_id],
				"utility": utilities[max_id]
				})

			doe_input = np.append(doe_input, [subset_points[max_id]], axis=0)
			doe_response = np.append(doe_response, [mean[max_id]])
			
			self.fantasy_model.train(doe_input, doe_response)
			mean, variance = self.fantasy_model.execute(subset_points, with_grad=False)

			#m, v = self.model.fantasize([[1,1],[2,2]], [10,5], [[1,1],[2,2]])
			#print(m,v)

		return np.array(batch)