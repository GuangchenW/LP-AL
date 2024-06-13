from .objective_function import BaseObjectiveFunction

import numpy as np

class G_Axle(BaseObjectiveFunction):
	def __init__(self):
		# An active learning Bayesian... Xiao et al. 2022
		super().__init__(name="front_axle", ndim=6)

	def _evaluate(self, x):
	    a,b,t,h,M,T = x

	    W_x = a*(h-2*t)**3/(6*h)+b/(6*h)*(h**3-(h-2*t)**3)
	    W_p = 0.8*b*t**2 + 0.4*(a**3*(h-2*t)/t)
	    sigma = M/W_x
	    tau = T/W_p
	    sigma_s = 460

	    return (sigma_s-np.sqrt(sigma**2+3*tau**2))

	def variable_definition(self):
		a = np.random.normal(12, 0.06)
		b = np.random.normal(65, 0.325)
		t = np.random.normal(14, 0.07)
		h = np.random.normal(85, 0.425)
		M = np.random.normal(3.5e6, 175000)
		T = np.random.normal(3.1e6, 155000)

		return [a,b,t,h,M,T]