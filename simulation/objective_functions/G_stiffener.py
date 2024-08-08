from .objective_function import BaseObjectiveFunction

import os
import matlab.engine
import numpy as np
from scipy.stats import norm

class G_Stiffener(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="Stiffener", ndim=10, failure_probability=0.04794)
		self.eng = matlab.engine.start_matlab()
		self.eng.cd(r"objective_functions/wing_stiffener", nargout=0)

	def _evaluate(self, x):
		E,d,P1,P2,F1,F2,F3,F4,F5,F6 = x

		response = self.eng.Simulate(E,d,P1,P2,F1,F2,F3,F4,F5,F6)

		return (0.0038-response)*100

	def variable_definition(self):
		# All variables are normal and have C.O.V. of 0.05, so common generator.
		def get_rn(mean):
			return np.random.normal(mean, mean*0.05)
		E = get_rn(100) # GPa
		d = get_rn(5) # mm
		P1 = get_rn(5000) # Pa
		P2 = get_rn(5000) # Pa
		F1 = get_rn(23758) # N
		F2 = get_rn(35239) # N
		F3 = get_rn(5949) # N
		F4 = get_rn(16245) # N
		F5 = get_rn(19185) # N
		F6 = get_rn(10140) # N

		return [E,d,P1,P2,F1,F2,F3,F4,F5,F6]

	def logpdf(self, x):
		E,d,P1,P2,F1,F2,F3,F4,F5,F6 = self.denormalize_data(x)
		# TODO: Could probably use loop for this with a mean array
		def get_prob(val, mean):
			return norm.logpdf(val, mean, mean*0.05)
		prob = get_prob(E, 100)
		prob += get_prob(d, 5)
		prob += get_prob(P1, 5000)
		prob += get_prob(P2, 5000)
		prob += get_prob(F1, 23758)
		prob += get_prob(F2, 35239)
		prob += get_prob(F3, 5949)
		prob += get_prob(F4, 16245)
		prob += get_prob(F5, 19185)
		prob += get_prob(F6, 10140)

		return prob