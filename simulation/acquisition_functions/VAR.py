import math
import numpy as np
from .acquisition_function import BaseAcquisitionFunction

class VAR(BaseAcquisitionFunction):
	def __init__(self, device="cpu", logger=None):
		super().__init__(name="VAR", device=device, logger=logger)

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = -variance
		
		# Down-shift the acquisiton values so max(acq(x))<=0.
		# This is done so it can be scaled with penalties for batching.
		max_val = np.max(acq)
		if max_val > 0:
			acq = acq - max_val - 1e-6

		return acq
