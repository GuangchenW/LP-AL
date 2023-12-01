import numpy as np

def U(candidates, mean, variance, doe_input, doe_response):
	return np.array(list(map(
			lambda pair: _U(pair[0], pair[1]), 
			zip(mean, variance)
			)))

def _U(mean, variance):
	return 1000 if variance < 0.1 else abs(mean)/np.sqrt(variance)


def ULP(candidates, mean, variance, doe_input, doe_response):
	dist = []
	util = []
	
	for i in range(len(candidates)):
		d,u = LF(candidates[i], mean[i], variance[i], doe_input, doe_response)
		dist.append(d)
		util.append(u)
	
	max_d = np.max(dist)

	def slerp(x):
		return 1/(1+np.exp(-10*(x-0.5)))

	out = []
	for d,u in zip(dist, util):
		out.append(u/slerp(d/max_d))
	return np.array(out)

def LF(candidate, mean, variance, doe_input, doe_response):
	if variance < 0.001:
		return [0, 1000]

	min_U = 1000
	min_d = np.inf
	for i in range(len(doe_input)):
		dist = np.linalg.norm(candidate - doe_input[i])
		#max_U = util if util > max_U else max_U
		#max_d = dist if dist > max_d else max_d
		if dist < min_d:
			min_d = dist
			min_U = ULP_helper(candidate, doe_response[i], mean, variance)
	return [min_d, min_U]

def ULP_helper(candidate, perform_near, mean, variance):
	denominator = np.sqrt((mean-perform_near)**2+variance)
	return abs(mean)/denominator