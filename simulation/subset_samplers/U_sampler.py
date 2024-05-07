import numpy as np

from .sampler import Sampler

class U_Sampler(Sampler):
	def __init__(self, threshold=1.96):
		super().__init__()
		self.threshold = threshold

	def sample(
		self, 
		mcs_population,
		doe_input, 
		doe_response,
		mean, 
		variance
	):
		if self.agressive:
			return mcs_population, mean, variance

		utilities = np.array([self._U(mu, var) for mu, var in zip(mean, variance)])

		indices = [i for i in range(len(utilities)) if utilities[i]<self.threshold]

		subset_pop = np.array([mcs_population[i] for i in indices])
		subset_mean = np.array([mean[i] for i in indices])
		subset_var = np.array([variance[i] for i in indices])

		return subset_pop, subset_mean, subset_var

	def _U(self, mean, variance):
		if variance < 1e-10:
			return float("nan")

		std = np.sqrt(variance)
		return abs(mean)/std