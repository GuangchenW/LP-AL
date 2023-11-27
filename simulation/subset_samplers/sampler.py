from abc import ABC, abstractmethod

class Sampler(ABC):
	def __init__(self):
		pass

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