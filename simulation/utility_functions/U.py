import numpy as np

def U(mean, variance):
	return 1000 if variance < 0.0001 else abs(mean)/np.sqrt(variance)