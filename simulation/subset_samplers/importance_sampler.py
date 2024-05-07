import numpy as np
from scipy.stats import norm

from .sampler import Sampler

class ImportanceSampler(Sampler):
	def __init__(self, n_samples=1000):
		super().__init__()
		self.nsamples = n_samples

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

		importances = np.array([self.importance(pnt, mu, var) for pnt, mu, var in zip(mcs_population, mean, variance)])
		total_importance = np.sum(importances)
		importances = importances/total_importance

		indices = [i for _,i in sorted(zip(importances, range(len(importances))))]
		#indices = self.pseudo_metro_hasting(importances, 1000)
		indices = indices[-1000:]

		subset_pop = np.array([mcs_population[i] for i in indices])
		subset_mean = np.array([mean[i] for i in indices])
		subset_var = np.array([variance[i] for i in indices])

		return subset_pop, subset_mean, subset_var

	def importance(self, pnt, mean, variance):
		if variance < 1e-10:
			return float("-inf")

		std = np.sqrt(variance)
		prob = np.exp(self.obj_func.logpdf(pnt))
		return norm.cdf(-abs(mean)/std)*prob

	def pseudo_metro_hasting(self, prob, n_samples):
		samples = []
		n_points = len(prob)
		idx = np.random.randint(n_points)
		samples.append(idx)

		for _ in range(n_samples*2):
			old_prob = prob[idx]
			_idx = np.random.randint(n_points)

			if prob[_idx] > old_prob:
				samples.append(_idx)
				idx = _idx
			elif np.random.uniform() < prob[_idx]-old_prob:
				samples.append(_idx)

			if len(samples) >= n_samples:
				break

		return np.array(samples)