from .objective_function import BaseObjectiveFunction

import numpy as np

class G_Beam(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="cantilever_beam", dim=3)

	def _evaluate(self, x):
		w=x[0]
		L=x[1]
		b=x[2]
		E=26
		I=(b**4)/12
		return (L/325-w*b*L**4/(8*E*I))*10

	def data_definition(self):
		w = np.random.normal(1000, 100)
		L = np.random.normal(6, 0.9)
		b = np.random.normal(250, 37.5)
		return [w,L,b]