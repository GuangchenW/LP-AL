import numpy as np

from .sampler import Sampler

class NaiveSampler(Sampler):
	def __init__(self):
		super().__init__()

	def sample(
		self, 
		mcs_population,
		doe_input, 
		doe_response,
		mean, 
		variance
	):
		return mcs_population, mean, variance