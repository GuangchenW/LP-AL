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
		
		# Up-shift the acquisiton values so min(acq(x))>=0.
		# This is done so it can be scaled with penalties for batching.
		min_val = np.min(acq)
		if min_val < 0:
			acq = acq - min_val

		return acq


	def _ULP(self, candidate, mean, variance, doe_input, doe_response):
		if variance < 1e-10:
			return float("nan")
		square_norm = ((doe_input-candidate)**2).sum()
		closest_id = np.argmin(square_norm)
		
		adjusted_var = np.sqrt((mean-doe_response[closest_id])**2+variance)
		adjusted_var = max(1e-4, adjusted_var)
		return -abs(mean)/adjusted_var
