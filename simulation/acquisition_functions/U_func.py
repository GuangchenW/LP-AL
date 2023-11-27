import numpy as np

from .acquisition_functions import AcquisitionFunction
from simulation.utility_functions import U

class U_Basic(AcquisitionFunction):
	def __init__(self, device="cpu"):
		self.name = "U"
		super().__init__(name=self.name, device=device)

	def acquire(
		self,
		model,
		input_population,
		doe_input,
		doe_response,
		n_points=1
	):
		mu, var = model.execute(input_population)

		utilities = np.array(list(map(
        	lambda pair: U(pair[0], pair[1]), 
        	zip(mu, var)
        	)))

		min_id = np.argmin(utilities)

		return {
		"next": input_population[min_id],
		"mean": mu[min_id],
		"variance": var[min_id],
		"utility": utilities[min_id]
		}