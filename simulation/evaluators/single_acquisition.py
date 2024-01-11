import numpy as np

from .acquisition_functions import AcquisitionFunction

class Single_Acquisition(AcquisitionFunction):
	def __init__(self, utility_func="U", device="cpu"):
		super().__init__(utility_func=utility_func, device=device)

	def acquire(
		self,
		input_population,
		doe_input,
		doe_response,
		mean,
		variance,
		n_points=1
	):
		utilities = self.utility_func(input_population, mean, variance, doe_input, doe_response)

		min_id = np.argmin(utilities)

		return np.array([{
		"next": input_population[min_id],
		"mean": mean[min_id],
		"variance": variance[min_id],
		"utility": utilities[min_id]
		}])