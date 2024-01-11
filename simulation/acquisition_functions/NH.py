import math
import numpy as np
from scipy import stats
from .acquisition_function import BaseAcquisitionFunction

class NH(BaseAcquisitionFunction):
	def __init__(self, device="cpu", logger=None):
		super().__init__(name="NH", device=device, logger=logger)

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = np.array([self._H(pnt, mu, var) for pnt, mu, var in zip(subset_points, mean, variance)])
		
		# Down-shift the acquisiton values so max(acq(x))<=0.
		# This is done so it can be scaled with penalties for batching.
		max_val = np.max(acq)
		if max_val > 0:
			acq = acq - max_val

		return acq


	def _H(self, candidate, mean, variance):
		# Since H is a maximization heuristic, we return the negative of EFF (NEFF).
		mu = mean
		var = variance

		std = np.sqrt(var)

		std = max(1e-4, std)

		upper = (2*std-mu)/std
		lower = (-2*std-mu)/std

		term1 = np.log(np.sqrt(2*np.pi)*std+0.5)*(stats.norm.cdf(upper)-stats.norm.cdf(lower))
		term2 = (std-0.5*mu)*stats.norm.pdf(upper)+(std+0.5*mu)*stats.norm.pdf(lower)
		h = abs(term1-term2)

		return -h