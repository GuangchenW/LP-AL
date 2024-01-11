import math
import numpy as np
from .acquisition_function import BaseAcquisitionFunction

class ULP(BaseAcquisitionFunction):
	def __init__(self, device="cpu", logger=None):
		super().__init__(name="ULP", device=device, logger=logger)

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = np.array([self._ULP(pnt, mu, var, doe_input, doe_response) for pnt, mu, var in zip(subset_points, mean, variance)])
		
		# Down-shift the acquisiton values so max(acq(x))<=0.
		# This is done so it can be scaled with penalties for batching.
		max_val = np.max(acq)
		if max_val > 0:
			acq = acq - max_val - 1e-6

		return acq


	def _ULP(self, candidate, mean, variance, doe_input, doe_response):
		square_norm = ((doe_input-candidate)**2).sum()
		closest_id = np.argmin(square_norm)
		
		adjusted_var = np.sqrt((mean-doe_response[closest_id])**2+variance)
		adjusted_var = max(1e-4, adjusted_var)
		return abs(mean)/adjusted_var
