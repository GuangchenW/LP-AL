from .objective_function import BaseObjectiveFunction

import numpy as np
from scipy.stats import norm, gumbel_r, uniform

class G_Oscillator(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="nonlinear_oscillator", ndim=6)
		self.failure_probability = 0.02865

	def _evaluate(self, x):
		c1,c2,m,r,t,F = x
		w_0 = np.sqrt((c1+c2)/m)
		val = 2*F*np.sin(w_0*t*0.5)/(m*w_0**2)
		return 3*r-abs(val)

	def data_definition(self):
		c1 = np.random.normal(1, 0.1)
		c2 = np.random.normal(0.1, 0.01)
		m = np.random.normal(1, 0.05)
		r = np.random.normal(0.5, 0.05)
		t = np.random.normal(1, 0.2)
		F = np.random.normal(1, 0.2)
		return [c1,c2,m,r,t,F]

	def logpdf(self, x):
		c1,c2,m,r,t,F = self.denormalize_data(x)
		prob = norm.logpdf(c1, 1, 0.1)
		prob += norm.logpdf(c2, 0.1, 0.01)
		prob += norm.logpdf(m, 1, 0.05)
		prob += norm.logpdf(r, 0.5, 0.05)
		prob += norm.logpdf(t, 1, 0.2)
		prob += norm.logpdf(F, 1, 0.2)

		return prob