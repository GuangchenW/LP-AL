from .objective_function import BaseObjectiveFunction

import numpy as np

class G_Ras(BaseObjectiveFunction):
	def __init__(self, d=5):
		super().__init__(name="Ras", dim=2)
		self.d = d

	def evaluate(self, x):
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