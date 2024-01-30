from .objective_function import BaseObjectiveFunction

import numpy as np

class G_Roof(BaseObjectiveFunction):
	def __init__(self):
		# An active learning Bayesian... Xiao et al. 2022
		super().__init__(name="roof_truss", dim=6)

	def _evaluate(self, x):
	    q,l,E_s,E_c,A_s,A_c = x

	    return 0.03-0.5*(q*l**2)*(3.81/(A_c*E_c)+1.13/(A_s*E_s))

	def data_definition(self):
		q = np.random.normal(2e4, 1600)
		l = np.random.normal(12, 0.24)
		E_s = np.random.normal(1.2e11, 8.4e9)
		E_c = np.random.normal(3e10, 2.4e9)
		A_s = np.random.uniform(9.3e-4, 9.5e-4)
		A_c = np.random.uniform(0.033, 0.035)

		return [q,l,E_s,E_c,A_s,A_c]