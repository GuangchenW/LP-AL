import math
import numpy as np
from scipy import stats
from .acquisition_function import BaseAcquisitionFunction

class EFF(BaseAcquisitionFunction):
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
		
		# Up-shift the acquisiton values so min(acq(x))>=0.
		# This is done so it can be scaled with penalties for batching.
		min_val = np.min(acq)
		if min_val < 0:
			acq = acq - min_val

		return acq


	def _NEFF(self, candidate, mean, variance):
		if variance < 1e-10:
			return float("nan")

		mu = mean
		var = variance
		std = np.sqrt(var)

		# HACK? disregard candidates with low variance as it likely
		# won't provide a lot of information.
		# Also avoids division by zero stuff.
		#if std < 0.05:
		#	return math.inf

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

		return eff