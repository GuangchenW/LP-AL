import numpy as np

def U(mean, variance):
	return np.array(list(map(
        	lambda pair: _U(pair[0], pair[1]), 
        	zip(mean, variance)
        	)))

def _U(mean, variance):
	return 1000 if variance < 0.0001 else abs(mean)/np.sqrt(variance)


def UPE(candidates, mean, variance, doe_input, doe_response):
		out = []
		for i in range(len(candidates)):
			out.append(LF(candidates[i], mean[i], variance[i], doe_input, doe_response))
		return np.array(out)

def LF2(candidate, mean, variance, doe):
    _U = U_orig_helper(mean, variance)
    min_d = np.inf
    for observation in doe:
        dist = np.linalg.norm(candidate - observation)
        min_d = dist if dist < min_d else min_d
    return _U+2/min_d

def LF(candidate, mean, variance, doe_input, doe_response):
    if variance < 0.0001:
        return 1000

    max_U = 0
    target_U = 0
    max_d = 0
    min_d = np.inf
    for i in range(len(doe_input)):
        dist = np.linalg.norm(candidate - doe_input[i])
        util = U_mod_helper(candidate, doe_response[i], mean, variance)
        max_U = util if util > max_U else max_U
        max_d = dist if dist > max_d else max_d
        if dist < min_d:
            min_d = dist
            target_U = util
    out = (target_U/max_U)/(min_d/max_d)
    return out

def U_mod_helper(candidate, perform_near, mean, variance):
    denominator = np.sqrt((mean-perform_near)**2+variance)
    return abs(mean)/denominator