import math
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from scipy.special import comb, factorial2
from .acquisition_function import BaseAcquisitionFunction

class LIF(BaseAcquisitionFunction):
	def __init__(self, obj_func, device="cpu", logger=None):
		super().__init__(name="LIF", device=device, logger=logger)
		self.obj_func =obj_func

	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		acq = np.array([self._LIF(pnt, mu, var, self.obj_func.dim, self.obj_func.logpdf(pnt)) for pnt, mu, var in zip(subset_points, mean, variance)])
		
		# Up-shift the acquisiton values so min(acq(x))>=0.
		# This is done so it can be scaled with penalties for batching.
		min_val = np.min(acq)
		if min_val < 0:
			acq = acq - min_val

		return acq


	def _LIF(self, candidate, mean, variance, M, logprob_cand):
		if variance < 1e-10:
			return float("nan")

		mu = mean
		var = variance
		std = np.sqrt(var)

		lif = norm.logcdf(-abs(mean)/std) + logprob_cand
		sum_term = 0
		if M % 2 == 0:
			sum_term += mean**M
			for m in range(1, M//2+1):
				sum_term += self.even_sum_term(M, m, mean, variance)
		else:
			for m in range(M):
				l = self.odd_sum_term(M, m, mean, variance)
				#if l > 1e40:
				#	print(l)
				sum_term += l
			sum_term *= np.sqrt(2/np.pi)

		lif = np.exp(lif)*sum_term

		return lif

	def even_sum_term(self, M, m, mean, variance):
		return comb(M, 2*m, exact=True)*mean**(M-2*m)*variance**m*factorial2(2*m-1)

	def odd_sum_term(self, M, m, mean, std):
		return comb(M, m)*mean**(M-m)*std**m*self.odd_integral(m, mean, std)

	def odd_integral(self, m, mean, std):
		if m == 0:
			return np.sqrt(2*np.pi)*(1-norm.cdf(-mean/std))
		if m == 1:
			return np.exp(-mean**2/(2*std**2))

		return (-mean/std)**(m-1)*np.exp(-mean**2/(2*std**2))+(m-1)*self.odd_integral(m-2, mean, std)
