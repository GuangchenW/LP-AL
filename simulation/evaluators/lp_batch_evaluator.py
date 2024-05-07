import sys
import math
import numpy as np
from scipy.stats import norm

from .base_evaluator import BaseEvaluator

class LP_Batch(BaseEvaluator):
	def __init__(self, acq_func, logger=None):
		super().__init__(acq_func=acq_func, logger=logger)
		self.L = float("nan")
		self.name = "lp"

	def set_grad(self, grad):
		self.grad = grad

	def set_L(self, L):
		self.L = max(0.25,L)
		print(self.L)

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
		utilities = self.soft_plus_transform(utilities)
		utilities = np.log(utilities)

		# Use 50 random samples to estimate L; reduce the effect of extreme gradients in Kriging estimation
		if len(self.grad) > 50:
			grad_samples = np.random.choice(len(self.grad), 50)
			grad_norm = np.linalg.norm(self.grad[grad_samples], axis=1)
		else:
			grad_norm = np.linalg.norm(self.grad, axis=1)
		self.set_L(grad_norm.max())

		for i in range(min(n_points, len(subset_points))):
			max_id = np.argmax(utilities)
			batch.append({
				"next": subset_points[max_id],
				"mean": mean[max_id],
				"variance": variance[max_id],
				"utility": utilities[max_id]
				})

			#self.set_L(np.linalg.norm(self.grad[max_id]))

			utilities = self.apply_hammer(subset_points, batch[-1], utilities)

		return np.array(batch)

	def soft_plus_transform(self, x):
		# With numpy 80 bits precision, for x > 100 the soft-plus transform doesn't do anything 
		# and may even fail due to large numbers. The soft-plus transform is mostly for dealing with 0
		# so this workaround should be fine, 
		return np.array([np.log(1+np.exp(_x)) if _x < 100 else _x for _x in x])

	def apply_hammer(self, candidates, batch, utilities):
		hammered_util = []
		for i in range(len(candidates)):
			hammer = self.log_hammer_func(candidates[i], batch["next"], batch["variance"], batch["mean"])
			util = utilities[i] + hammer
			hammered_util.append(util)
		return np.array(hammered_util)

	def log_hammer_func(self, candidate, center, variance, offset):
		dist = np.linalg.norm(center-candidate)
		z = (self.L*dist-abs(offset))/np.sqrt(2*variance)
		phi = 0.5*math.erfc(-z)
		phi = max(sys.float_info.min, phi) # Prevent cases where erfc evaluates to 0
		return np.log(phi)

	def alt_log_hammer_func(self, candidate, center, variance, offset):
		new_offset = np.sqrt(2/np.pi)*np.sqrt(variance)*np.exp(-offset**2/(2*variance))+offset*(1-2*norm.cdf(-offset/np.sqrt(variance)))
		new_variance = offset**2+variance-new_offset**2
		dist = np.linalg.norm(center-candidate)
		z = (self.L*dist-abs(new_offset))/np.sqrt(2*new_variance)
		phi = 0.5*math.erfc(-z)
		phi = max(sys.float_info.min, phi) # Prevent cases where erfc evaluates to 0
		return np.log(phi)
