import math
import torch
import gpytorch
import numpy as np

from gpytorch.models import ExactGP
from gpytorch.means import Mean, LinearMean, ConstantMean, ZeroMean
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import matplotlib.pyplot as plt

class _ExactGP(ExactGP):
	def __init__(self, train_x, train_y, likelihood, mean_func, covar_kernel):
		super(_ExactGP, self).__init__(train_x, train_y, likelihood)
		self.mean_module = mean_func
		self.covar_kernel = covar_kernel

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_kernel(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRegression:
	def __init__(self, n_dim, mean_func=None, covar_kernel=None, logger=None, silent=True):

		self.mean_func = mean_func or ConstantMean()

		if covar_kernel is None:
			self.base_covar_kernel = RBFKernel(ard_num_dims=n_dim)
			self.covar_kernel = ScaleKernel(self.base_covar_kernel)
		else:
			self.covar_kernel = covar_kernel

		self.logger = logger
		self.silent = silent
		self.gp = None

		self.normalize = False

	def normalize_inputs(self, inputs):
		inputs = torch.tensor(inputs, dtype=torch.double)
		self.train_mean = inputs.mean(dim=0)
		self.train_std = inputs.std(dim=0)
		normalized_inputs = (inputs-self.train_mean)/self.train_std
		return normalized_inputs

	def normalize_targets(self, targets):
		targets = torch.tensor(targets, dtype=torch.double)
		if not self.normalize:
			self.target_mean = 0
			self.target_std = 1
			return targets
		self.target_mean = targets.mean(dim=0)
		self.target_std = targets.std(dim=0)
		normalized_targets = (targets-self.target_mean)/self.target_std
		return normalized_targets

	def train(self, inputs, targets, device="cpu"):
		normalized_targets = self.normalize_targets(targets)
		#normalized_targets = torch.tensor(targets, dtype=torch.double)
		inputs = torch.tensor(inputs, dtype=torch.double)

		if self.gp == None:
			self.noise = torch.ones(inputs.shape[0])*1e-4
			self.likelihood = FixedNoiseGaussianLikelihood(noise=self.noise)
			self.gp = _ExactGP(inputs, normalized_targets, self.likelihood, self.mean_func, self.covar_kernel)
			self.gp.to(device)
		else:
			self.gp.set_train_data(inputs, normalized_targets, False)
			self.noise = torch.ones(inputs.shape[0])*1e-4
			self.gp.likelihood.noise = self.noise
		self.gp.train()
		self.gp.likelihood.train()

		optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)

		mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

		max_iter = 999
		epsilon = 1e-4
		loss_history = []
		for i in range(max_iter):
			optimizer.zero_grad()
			output = self.gp(inputs)
			loss = -mll(output, normalized_targets)
			loss.backward()
			if not self.silent:
				print("Iter %d - Loss: %.3f" % (i+1, loss.item()))

			optimizer.step()

			loss_history.append(loss.item())
			if len(loss_history) > 3 and np.abs(np.diff(loss_history[-5:])).max() < epsilon:
				break

			#if abs(prev_loss-loss.item()) < epsilon:
			
		self.gp.eval()
		self.likelihood.eval()

		if not self.silent:
			ls = self.base_covar_kernel.lengthscale
			print(f"Lengthscale:{ls}")

	def execute(self, inputs, with_grad=False):
		inputs = torch.tensor(inputs, dtype=torch.double)
		
		if with_grad:
			inputs.requires_grad = True

			pred = self.likelihood(self.gp(inputs), noise=torch.zeros(inputs.shape[0]))

			mean = pred.mean.detach()
			mean = mean * self.target_std + self.target_mean

			mean_pred = pred.mean.sum()
			mean_pred.backward(retain_graph=True)
			#grad_mean = inputs.grad/self.train_std*self.target_std
			grad_mean = inputs.grad*self.target_std
			#print("INFO", grad_mean)

			#TODO
			#min_lipschitz = 0.25
			#lipschitz = max(torch.max(torch.norm(grad_mean, dim=1)).item(), min_lipschitz)

			return (mean.numpy(), pred.variance.detach().numpy(), grad_mean.detach().numpy())
		else:
			with torch.no_grad(), gpytorch.settings.fast_pred_var():
				f_preds = self.gp(inputs)
				mean = f_preds.mean * self.target_std + self.target_mean
				return (mean.numpy(), f_preds.variance.numpy())

	# Obsolete, do not use
	def fantasize(self, inputs, targets, tests):
		noises = torch.ones(np.shape(inputs)[0])*0.001
		inputs = torch.tensor(inputs, dtype=torch.double)
		targets = torch.tensor(targets, dtype=torch.double)
		tests = torch.tensor(tests, dtype=torch.double)
		model = self.gp.get_fantasy_model(inputs, targets, noise=noises)
		with torch.no_grad():
			f_preds = model(tests)
			return (f_preds.mean.numpy(), f_preds.variance.numpy())

if __name__ == "__main__":
	train_x = torch.linspace(0,1,100)
	train_y = torch.sin(train_x*2*math.pi+torch.randn(train_x.size())*math.sqrt(0.04))
	train_x = train_x.numpy()
	train_y = train_y.numpy()
	test = GPRegression(n_dim=1, silent=False)
	normal_x = test.normalize_inputs(train_x)
	test.train(train_x, train_y)

	test_x = torch.tensor([0.25])
	f_preds = test.gp(test_x)
	print(f_preds.mean)