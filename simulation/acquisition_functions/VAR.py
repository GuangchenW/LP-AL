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
		acq = variance
		
		# Up-shift the acquisiton values so min(acq(x))>=0.
		# This is done so it can be scaled with penalties for batching.
		min_val = np.min(acq)
		if min_val < 0:
			acq = acq - min_val

		return acq
