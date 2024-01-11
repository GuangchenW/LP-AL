import math
import numpy as np
from scipy import stats
from .acquisition_function import BaseAcquisitionFunction

class NEFF(BaseAcquisitionFunction):
	def __init__(self, device="cpu", logger=None):
		super().__init__(name="NEFF", device=device, logger=logger)

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = np.array([self._NEFF(pnt, mu, var) for pnt, mu, var in zip(subset_points, mean, variance)])
		
		# Down-shift the acquisiton values so max(acq(x))<=0.
		# This is done so it can be scaled with penalties for batching.
		max_val = np.max(acq)
		if max_val > 0:
			acq = acq - max_val

		return acq


	def _NEFF(self, candidate, mean, variance):
		# Since EFF is a maximization heuristic, we return the negative of EFF (NEFF).
		mu = mean
		var = variance
		std = np.sqrt(var)

		# HACK? disregard candidates with low variance as it likely
		# won't provide a lot of information.
		# Also avoids division by zero stuff.
		#if std < 0.05:
		#	return math.inf

		std = max(1e-4, std)

		epsilon = 2*std
		cprob = stats.norm.cdf(-mu/std)
		cprob_low = stats.norm.cdf((-epsilon-mu)/std)
		cprob_high = stats.norm.cdf((epsilon-mu)/std)

		term1=mu*(2*cprob-cprob_low-cprob_high)

		prob = stats.norm.pdf(-mu/std)
		prob_low = stats.norm.pdf((-epsilon-mu)/std)
		prob_high = stats.norm.pdf((epsilon-mu)/std)
		
		term2=std*(2*prob-prob_low-prob_high)
		term3=cprob_high-cprob_low

		eff = term1-term2+term3

		return -eff