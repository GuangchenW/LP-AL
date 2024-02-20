import math
import numpy as np
from scipy import stats
from .acquisition_function import BaseAcquisitionFunction

class ERF(BaseAcquisitionFunction):
	def __init__(self, device="cpu", logger=None):
		super().__init__(name="ERF", device=device, logger=logger)

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = np.array([self._ERF(pnt, mu, var) for pnt, mu, var in zip(subset_points, mean, variance)])
		
		# Up-shift the acquisiton values so min(acq(x))>=0.
		# This is done so it can be scaled with penalties for batching.
		min_val = np.min(acq)
		if min_val < 0:
			acq = acq - min_val

		return acq


	def _ERF(self, candidate, mean, variance):
		if variance < 1e-10:
			return float("nan")

		mu = mean
		var = variance
		std = np.sqrt(var)


		term1 = -abs(mean)*stats.norm.cdf(-abs(mean)/std)
		
		term2=std*stats.norm.pdf(mean/std)

		erf = term1+term2

		return erf