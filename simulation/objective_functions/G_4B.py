from .objective_function import BaseObjectiveFunction

import numpy as np
from scipy.stats import norm

class G_4B(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="4B", dim=2)
		self.k = 7
		self.failure_probability = 0.002230

	def _evaluate(self, x):
		x1=x[0]
		x2=x[1]
		b1 = 3 + 0.1*(x1-x2)**2 - (x1+x2)/np.sqrt(2)
		b2 = 3 + 0.1*(x1-x2)**2 + (x1+x2)/np.sqrt(2)
		b3 = (x1-x2) + self.k/np.sqrt(2)
		b4 = (x2-x1) + self.k/np.sqrt(2)
		return np.min([b1, b2, b3, b4])

	def data_definition(self):
		x1 = np.random.normal(0, 1)
		x2 = np.random.normal(0, 1)

		return [x1, x2]

	def logpdf(self, x):
		x1, x2 = self.denormalize_data(x)
		prob = norm.logpdf(x1, 0, 1)
		prob += norm.logpdf(x2, 0, 1)

		return prob