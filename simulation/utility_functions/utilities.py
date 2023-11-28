import numpy as np

def U(candidates, mean, variance, doe_input, doe_response):
	return np.array(list(map(
			lambda pair: _U(pair[0], pair[1]), 
			zip(mean, variance)
			)))

def _U(mean, variance):
	return 1000 if variance < 0.0001 else abs(mean)/np.sqrt(variance)


def ULP(candidates, mean, variance, doe_input, doe_response):
	dist = []
	util = []
	
	for i in range(len(candidates)):
		d,u = LF(candidates[i], mean[i], variance[i], doe_input, doe_response)
		dist.append(d)
		util.append(u)
	
	max_d = np.max(dist)
	max_u = np.max(util)

	out = []
	for d,u in zip(dist, util):
		out.append((u/max_u)/(d/max_d))
	return np.array(out)

def LF(candidate, mean, variance, doe_input, doe_response):
	if variance < 0.0001:
		return 1000

	max_U = 0
	min_U = 0
	max_d = 0
	min_d = np.inf
	for i in range(len(doe_input)):
		dist = np.linalg.norm(candidate - doe_input[i])
		util = ULP_helper(candidate, doe_response[i], mean, variance)
		#max_U = util if util > max_U else max_U
		#max_d = dist if dist > max_d else max_d
		if dist < min_d:
			min_d = dist
			min_U = util
	#out = (target_U/max_U)/(min_d/max_d)
	return [min_d, min_U]

def ULP_helper(candidate, perform_near, mean, variance):
	denominator = np.sqrt((mean-perform_near)**2+variance)
	return abs(mean)/denominator