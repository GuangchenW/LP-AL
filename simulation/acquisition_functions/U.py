import math
import numpy as np
from .acquisition_function import BaseAcquisitionFunction

class U(BaseAcquisitionFunction):
	def __init__(self, device="cpu", logger=None):
		super().__init__(name="U", device=device, logger=logger)

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = np.array([self._U(mu, var) for mu, var in zip(mean, variance)])
		
		# Up-shift the acquisiton values so min(acq(x))>=0.
		# This is done so it can be scaled with penalties for batching.
		min_val = np.min(acq)
		if min_val < 0:
			acq = acq - min_val
		return acq

	def _U(self, mean, variance):
		std = max(1e-4, np.sqrt(variance))
		return -abs(mean)/std
