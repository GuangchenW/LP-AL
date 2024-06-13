from .objective_function import BaseObjectiveFunction

import numpy as np

class G_Simple(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="simple", dim=2)

	def _evaluate(self, x):
		x1=x[0]
		x2=x[1]

		return x1**2+x2**2-2

	def variable_definition(self):
		x1 = np.random.normal(0, 1)
		x2 = np.random.normal(0, 1)

		return [x1, x2]