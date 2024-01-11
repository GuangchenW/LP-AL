import math
import numpy as np
from scipy import stats

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
	print("max distance : ", max_d)

	def slerp(x):
		return 1/(1+np.exp(-10*(x-0.5)))

	out = []
	for d,u in zip(dist, util):
		out.append(u/slerp(d/max_d))
	return np.array(out)

def LF(candidate, mean, variance, doe_input, doe_response):
	if variance < 0.0001:
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

def NEFF(candidates, mean, variance, doe_input, doe_response):
	dist = []
	util = []
	for i in range(len(candidates)):
		d,u = _EFF(candidates[i], mean[i], variance[i], doe_input, doe_response)
		dist.append(d)
		util.append(-u)

	max_d = np.max(dist)

	def slerp(x):
		return 1/(1+np.exp(-10*(x-0.5)))

	out = []
	for d,u in zip(dist, util):
		out.append(u/slerp(d/max_d))
	#return np.array(out)
	return util

def _EFF(candidate, mean, variance, doe_input, doe_response):
	mu = mean
	var = variance

	std = np.sqrt(var)

	if std < 0.05:
		return [0, -1000]

	epsilon = 2*std
	cprob = stats.norm.cdf(-mu/std)
	cprob_low = stats.norm.cdf((-epsilon-mu)/std)
	cprob_high = stats.norm.cdf((epsilon-mu)/std)

	term1=mu*(2*cprob-cprob_low-cprob_high)

	prob = stats.norm.pdf(-mu/std)
	prob_low = stats.norm.pdf((-epsilon-mu)/std)
	prob_high = stats.norm.pdf((epsilon-mu)/std)
	
	term2=std*(2*prob-prob_low-prob_high)
	term3=cprob_high-cprob_low

	eff = term1-term2+term3

	min_d = np.inf
	for i in range(len(doe_input)):
		dist = np.linalg.norm(candidate - doe_input[i])
		if dist < min_d:
			min_d = dist
		z = 1/0.15*dist+abs(doe_response[i])
		phi = 0.5*math.erfc(-z)
		eff *= phi

	return [min_d, eff]

def NH(candidates, mean, variance, doe_input, doe_response):
	dist = []
	util = []
	for i in range(len(candidates)):
		d,u = _H(candidates[i], mean[i], variance[i], doe_input, doe_response)
		dist.append(d)
		util.append(-u)
	return util

	max_d = np.max(dist)

	def slerp(x):
		return 1/(1+np.exp(-10*(x-0.5)))

	out = []
	for d,u in zip(dist, util):
		out.append(u/slerp(d/max_d))
	return np.array(out)

def _H(candidate, mean, variance, doe_input, doe_response):
	mu = mean
	var = variance

	std = np.sqrt(var)

	if std < 0.05:
		return [0, -1000]

	upper = (2*std-mu)/std
	lower = (-2*std-mu)/std

	term1 = np.log(np.sqrt(2*np.pi)*std+0.5)*(stats.norm.cdf(upper)-stats.norm.cdf(lower))
	term2 = (std-0.5*mu)*stats.norm.pdf(upper)+(std+0.5*mu)*stats.norm.pdf(lower)
	h = abs(term1-term2)

	min_d = np.inf
	for i in range(len(doe_input)):
		dist = np.linalg.norm(candidate - doe_input[i])
		if dist < min_d:
			min_d = dist

	return [min_d, h]