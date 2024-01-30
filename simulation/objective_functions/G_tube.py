from .objective_function import BaseObjectiveFunction

import numpy as np

class G_Tube(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="cantilever_tube", dim=9)

	def _evaluate(self, x):
	    t,d,L1,L2,F1,F2,P,T,S_y = x
	    theta1 = 0.08726646259971647 # 5 degrees
	    theta2 = 0.17453292519943295 # 10 degrees

	    M = F1*L1*np.cos(theta1)+F2*L2*np.cos(theta2)
	    A = np.pi/4*(d**2-(d-2*t)**2)
	    c = 0.5*d
	    I = np.pi/64*(d**4-(d-2*t)**4)
	    J = 2*I
	    tau = (T*d)/(2*J)
	    sigma_x = (P+F1*np.sin(theta1)+F2*np.sin(theta2))/A + M*c/I
	    sigma_max = np.sqrt(sigma_x**2+3*tau**2)

	    return (S_y - sigma_max)*0.1

	def data_definition(self):
		t = np.random.normal(5, 0.1)
		d = np.random.normal(42, 0.5)
		L1 = np.random.uniform(119.75, 120.25)
		L2 = np.random.uniform(59.75, 60.25)
		F1 = np.random.normal(3000, 300)
		F2 = np.random.normal(3000, 300)
		P = np.random.gumbel(30000, 3000)
		T = np.random.normal(90000, 9000)
		S_y = np.random.normal(220, 22)
		return [t,d,L1,L2,F1,F2,P,T,S_y]