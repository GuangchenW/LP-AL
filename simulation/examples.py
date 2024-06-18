import os

import numpy as np

from simulation import AKMCS
from objective_functions import AskTellFunction, AnalyticalFunction
from objective_functions import G_Simple, G_4B, G_Ras, G_Oscillator, G_Tube, G_FEM
from acquisition_functions import U, EFF, ERF, H, LIF
from evaluators import LP_Batch

def ask_tell_experiment():
	# Two variables following the standard normal distribution
	# lambda function returning a numpy array 
	var_def = lambda : [np.random.normal(0, 1), np.random.normal(0, 1)]

	# Joint log pdf of the input variables
	def var_logpdf(x):
		x1, x2 = x
		prob = norm.logpdf(x1, 0, 1)
		prob += norm.logpdf(x2, 0, 1)
		return prob

	# Make our limit state function object
	# Note the `name` parameter will be used to save/load the samples for Monte-Carlo simulation
	func = AskTellFunction(name="TestAskTell", ndim=2, variable_definition=var_def, variable_logpdf=var_logpdf, failure_probability=0.25)

	# Create our LP-AL instance, using the U learning function and batch size 8
	solver = AKMCS(acq_func=U(), batch_size=8)

	# Initialize with the limit state function
	solver.initialize_input(func, sample_size=10**4, seed=1)

	# We can't do MCS with ask-tell, it will be too slow. 
	# So set `do_mcs` to False (its default value) and use the `failure_probability` in the limit-state function
	result = solver.kriging_estimate(do_mcs=False)
	print(result)
	solver.visualize(mk_sample_pool_anim=True, save_visual=True, filename=["asktell.png", "asktell.gif"])

def analytical_experiment():
	# Two variables following the standard normal distribution
	# lambda function returning a numpy array 
	var_def = lambda : [np.random.normal(0, 1), np.random.normal(0, 1)]

	# Limit state function
	# Returns the max of the two variable. So the probability of failure should be 0.25 (third quadrant)
	def limit_state_func(x):
		x1,x2 = x
		return max(x1, x2)

	# Joint log pdf of the input variables
	def var_logpdf(x):
		x1, x2 = x
		prob = norm.logpdf(x1, 0, 1)
		prob += norm.logpdf(x2, 0, 1)
		return prob

	# Make our limit state function object
	# Note the `name` parameter will be used to save/load the samples for Monte-Carlo simulation
	# We can still give the function a precomputed probability of failure, but it's not required
	func = AnalyticalFunction(name="TestAnalytical", ndim=2, variable_definition=var_def, variable_logpdf=var_logpdf, limit_state_function=limit_state_func)

	# Create our LP-AL instance, using the U learning function and batch size 8
	solver = AKMCS(acq_func=U(), batch_size=8)

	# Initialize with the limit state function
	solver.initialize_input(func, sample_size=10**4, seed=1)

	# We will make the solver do MCS since the limit-state function is given in closed form
	result = solver.kriging_estimate(do_mcs=True)
	print(result)
	solver.visualize(mk_sample_pool_anim=True, save_visual=True, filename=["analytical.png", "analytical.gif"])

if __name__ == "__main__":
	#ask_tell_experiment()
	analytical_experiment()

