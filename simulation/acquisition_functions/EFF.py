import math
import numpy as np
from scipy.stats import norm
from .acquisition_function import BaseAcquisitionFunction

class EFF(BaseAcquisitionFunction):
	def __init__(self, device="cpu", logger=None):
		super().__init__(name="EFF", device=device, logger=logger)

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = np.array([self._EFF(mu, var) for mu, var in zip(mean, variance)])
		
		# Up-shift the acquisiton values so min(acq(x))>=0.
		# This is done so it can be scaled with penalties for batching.
		min_val = np.min(acq)
		if min_val < 0:
			acq = acq - min_val

		return acq

	def _EFF(self, mean, variance):
		if variance < 1e-10:
			return float("nan")

		mu = mean
		var = variance
		std = np.sqrt(var)

		epsilon = 2*std
		cprob = norm.cdf(-mu/std)
		cprob_low = norm.cdf((-epsilon-mu)/std)
		cprob_high = norm.cdf((epsilon-mu)/std)

		term1=mu*(2*cprob-cprob_low-cprob_high)

		prob = norm.pdf(-mu/std)
		prob_low = norm.pdf((-epsilon-mu)/std)
		prob_high = norm.pdf((epsilon-mu)/std)
		
		term2=std*(2*prob-prob_low-prob_high)
		term3=epsilon*(cprob_high-cprob_low)

		eff = term1-term2+term3

		return eff