from .objective_function import BaseObjectiveFunction

import numpy as np

class G_FEM(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="FEM", dim=6)

	def _evaluate(self, x):
	    P1,P2,P3,L,A,E = x
	    print("0:%f:%f,[0,%f]"%(L,L*2,L))
	    print("2 0 %f" % -P1)
	    print("3 %f %f" % (P2,-P3))
	    response = input("Enter y-displacement of node 3: ")

	    return 3.6-float(response)

	def data_definition(self):
		P1 = np.random.normal(80, 4)
		P2 = np.random.normal(10, 0.5)
		P3 = np.random.normal(10, 0.5)
		L = np.random.normal(1, 0.05)
		A = np.random.normal(10, 0.5)
		E = np.random.normal(100, 5)

		return [P1,P2,P3,L,A,E]