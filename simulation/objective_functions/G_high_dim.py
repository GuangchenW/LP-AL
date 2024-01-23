from .objective_function import BaseObjectiveFunction

import numpy as np

class G_High_Dim(BaseObjectiveFunction):
	def __init__(self):
		# An improved AK-MCS... Liu et al. 2016
		super().__init__(name="high_dimensional", dim=20)

	def evaluate(self, x):
		term_sum = x.sum()

		return self.dim+0.6*np.sqrt(self.dim)-term_sum

	def data_definition(self):
		data = []

		for i in range(self.dim):
			data.append(np.random.lognormal(-0.01961, 0.19804))

		return data