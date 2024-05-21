from objective_functions import G_Oscillator

import math, sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

from objective_functions import G_4B, G_Ras, G_Oscillator, G_Tube
from models import OrdinaryKriging
from acquisition_functions import U, EFF

def example_func(x):
	return np.sum(np.sin(x-1)+np.sin(3*x-3)+1, axis=1)

def example_func2(x):
	return np.sum(np.cos(2*x)+np.sin(x+1), axis=1)

def calc_U(mean, variance):
	if variance < 1e-7:
		return 0
	else:
		return abs(mean)/np.sqrt(variance)


def apply_hammer(candidates, center, center_mean, center_variance, L, utilities):
	hammered_util = []
	L = max(0.25, L)
	for i in range(len(candidates)):
		hammer = hammer_func(candidates[i], center, center_variance, center_mean, L)
		util = utilities[i] * hammer
		hammered_util.append(util)
	return np.array(hammered_util)

def hammer_func(candidate, center, variance, offset, L):
	dist = np.linalg.norm(center-candidate)
	z = (L*dist-abs(offset))/np.sqrt(2*variance)
	phi = 0.5*math.erfc(-z)
	phi = max(sys.float_info.min, phi) # Prevent cases where erfc evaluates to 0
	return phi

def soft_plus_transform(x):
		return np.array([np.log(1+np.exp(_x)) if _x < 100 else _x for _x in x])

if __name__ == "__main__":
	matplotlib.rcParams["mathtext.fontset"]="cm"
	model = OrdinaryKriging(n_dim=1)
	x = np.linspace(-3, 3, 600)[:,None]
	np.random.seed(22)
	test_inputs = np.random.uniform(-3, 3, (6,1))
	test_outputs = example_func(test_inputs)

	model.train(test_inputs, test_outputs)

	mean, variance, grad = model.execute(x, with_grad=True)

	acq_func = EFF()
	util = acq_func.acquire(x, mean, variance, test_inputs, test_outputs)
	new_util = util

	x = x.flatten()
	idx = np.nanargmax(new_util)

	fig, ax = plt.subplots()
	prev_score = "\\alpha(x;\\mathcal{B}_0)"
	ax.plot(x, mean, alpha=0.4, label="$\\hat{f}(x)$")
	ax.fill_between(x, mean-variance, mean+variance, alpha=0.1)
	ax.plot(x, new_util, color="red", label="$%s$"%prev_score)
	ax.plot(x[idx], new_util[idx], "*", markersize=10)
	ax.set_xlabel("$x$")
	ax.set_ylabel("value")
	ax.legend(fontsize="large", loc=2)
	fig.tight_layout()
	plt.savefig("0.pdf", format="pdf")

	L = np.absolute(grad).max()
	for i in range(2):

		fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios":[1,2]})

		hammer = np.array([hammer_func(c, x[idx], variance[idx], mean[idx], L) for c in x])
		penalty_name = "\\psi(x;x_{1,%d})"%(i+1)
		ax1.plot(x, hammer, color="green", label="$%s$"%penalty_name)
		ax1.legend(fontsize="large", loc=4)
		ax1.set_ylabel("value")

		ax2.plot(x, mean, alpha=0.4, label="$\\hat{f}(x)$")
		ax2.fill_between(x, mean-variance, mean+variance, alpha=0.1)
		# Previous utility
		# ax2.plot(x, new_util, color="red", label="$%s$"%prev_score, ls="--")
		# Apply penalty
		new_util = apply_hammer(x, x[idx], mean[idx], variance[idx], L, new_util)
		prev_score = prev_score+penalty_name
		ax2.plot(x, new_util, color="red", label="$%s$"%prev_score)

		idx = np.nanargmax(new_util)
		ax2.plot(x[idx], new_util[idx], "*", markersize=10)

		ax2.legend(fontsize="large", loc=2)
		ax2.set_xlabel("$x$")
		ax2.set_ylabel("value")
	
		fig.tight_layout()
		plt.savefig("%d.pdf"%(i+1), format="pdf")
