import numpy as np

from .sampler import Sampler
from simulation.acquisition_functions import U

class U_Sampler(Sampler):
	def __init__(self, threshold=1.96):
		super().__init__()
		self.threshold = threshold
		self.util_func = U()

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
		std = max(1e-4, np.sqrt(variance))
		return abs(mean)/std