from .objective_function import BaseObjectiveFunction

import numpy as np
from scipy.stats import norm

class G_Ras(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="modified_rastrigin", ndim=2)
		self.d = 5
		self.failure_probability = 0.2941

	def _evaluate(self, x):
		x1=x[0]
		x2=x[1]
		def calc_term(x_i):
			return x_i**2 - 5*np.cos(2*np.pi*x_i)
		term_sum = calc_term(x1) + calc_term(x2)
		result = self.d - term_sum
		return result

	def data_definition(self):
		x1 = np.random.normal(0, 1)
		x2 = np.random.normal(0, 1)

		return [x1, x2]

	def logpdf(self, x):
		x1, x2 = self.denormalize_data(x)
		prob = norm.logpdf(x1, 0, 1)
		prob += norm.logpdf(x2, 0, 1)

		return prob