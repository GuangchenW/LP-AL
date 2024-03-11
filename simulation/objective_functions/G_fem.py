from .objective_function import BaseObjectiveFunction

import os
import numpy as np
from oct2py import octave

class G_FEM(BaseObjectiveFunction):
	def __init__(self):
		super().__init__(name="FEM", dim=6)
		octave.addpath(os.path.dirname(os.path.realpath(__file__)))

	def _evaluate(self, x):
	    P1,P2,P3,L,A,E = x
	    # The E here is in kPa, so we need to apply a factor of 1000.
	    response = octave.eval("Truss2DBare(%f,%f,%f,%f,%f,%f)"%(P1, P2, P3, L, A, E*1000), verbose=False)

	    return 3.6+response

	def data_definition(self):
		P1 = np.random.normal(80, 4)
		P2 = np.random.normal(10, 0.5)
		P3 = np.random.normal(10, 0.5)
		L = np.random.normal(1, 0.05)
		A = np.random.normal(10, 0.5)
		E = np.random.normal(100, 5)

		return [P1,P2,P3,L,A,E]