from abc import ABC, abstractmethod

class Sampler(ABC):
	def __init__(self):
		self.agressive = False

	@abstractmethod
	def sample(self, 
		mcs_population,
		doe_input, 
		doe_response,
		mean, 
		variance
	):
		"""
		:param mcs_population: the monte-carlo population
		:param doe_input: all inputs acquired so far, including initial doe
		:param doe_response: outputs observed for `doe_input`
		:param mean: the estimated mean of `mcs_population`
		:param varaince: the estimated variance of `mcs_population`
		"""

	def aggressive_mode(self, should_be):
		self.agressive = should_be