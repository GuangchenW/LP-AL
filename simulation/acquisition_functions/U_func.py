import numpy as np

from .acquisition_functions import AcquisitionFunction
from simulation.utility_functions import U, UPE

class U_Basic(AcquisitionFunction):
	def __init__(self, device="cpu"):
		self.name = "U"
		super().__init__(name=self.name, device=device)

	def acquire(
		self,
		input_population,
		doe_input,
		doe_response,
		mean,
		variance,
		n_points=1
	):
		utilities = UPE(input_population, mean, variance, doe_input, doe_response)

		min_id = np.argmin(utilities)

		return {
		"next": input_population[min_id],
		"mean": mean[min_id],
		"variance": variance[min_id],
		"utility": utilities[min_id]
		}