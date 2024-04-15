from .objective_function import BaseObjectiveFunction

import numpy as np

class G_Oscillator(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="nonlinear_oscillator", dim=6)
		self.failure_probability = 0.028650

	def _evaluate(self, x):
		w_0 = np.sqrt((x[0]+x[1])/x[2])
		val = 2*x[5]*np.sin(w_0*x[4]*0.5)/(x[2]*w_0**2)
		return 3*x[3]-abs(val)

	def data_definition(self):
		c1 = np.random.normal(1, 0.1)
		c2 = np.random.normal(0.1, 0.01)
		m = np.random.normal(1, 0.05)
		r = np.random.normal(0.5, 0.05)
		t = np.random.normal(1, 0.2)
		F = np.random.normal(1, 0.2)
		return [c1,c2,m,r,t,F]