import math
import torch
import gpytorch
import numpy as np

from gpytorch.models import ExactGP
from gpytorch.means import Mean, ConstantMean, ZeroMean
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import matplotlib.pyplot as plt

class _ExactGP(ExactGP):
	def __init__(self, train_x, train_y, likelihood, mean_func, covar_kernel):
		super(_ExactGP, self).__init__(train_x, train_y, likelihood)
		self.mean_module = mean_func
		self.covar_kernel = covar_kernel
		#self.normalizer = torch.nn.BatchNorm1d(9, affine=False)

	def forward(self, x):
		#x = self.normalizer(x)
		mean_x = self.mean_module(x)
		covar_x = self.covar_kernel(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class OrdinaryKriging:
	def __init__(self, n_dim, mean_func=None, covar_kernel=None, init_lengthscale=None):

		self.mean_func = mean_func or ConstantMean()

		if covar_kernel is None:
			self.base_covar_kernel = RBFKernel(ard_num_dims=n_dim)
			if not init_lengthscale == None:
				self.base_covar_kernel.lengthscale = init_lengthscale
			self.covar_kernel = ScaleKernel(self.base_covar_kernel)
		else:
			self.covar_kernel = covar_kernel

		self.gp = None

	def train(self, inputs, labels, device="cpu"):
		inputs = torch.tensor(inputs, dtype=torch.double)
		self.train_mean = inputs.mean(dim=0)
		self.train_std = inputs.std(dim=0)
		normalized_inputs = (inputs-self.train_mean)/self.train_std
		labels = torch.tensor(labels, dtype=torch.double)

		if self.gp == None:
			self.noise = torch.ones(np.shape(inputs)[0])*0.001
			self.likelihood = FixedNoiseGaussianLikelihood(noise=self.noise)
			self.gp = _ExactGP(normalized_inputs, labels, self.likelihood, self.mean_func, self.covar_kernel)
			self.gp.to(device)
		else:
			self.gp.set_train_data(normalized_inputs, labels, False)
			self.noise = torch.ones(np.shape(inputs)[0])*0.001
			self.gp.likelihood.noise = self.noise
		self.gp.train()
		self.gp.likelihood.train()

		optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)

		mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

		max_iter = 2000
		epsilon = 0.001
		prev_loss = 100
		for i in range(max_iter):
			optimizer.zero_grad()
			output = self.gp(normalized_inputs)
			loss = -mll(output, labels)
			loss.backward()
			optimizer.step()

			if abs(prev_loss-loss.item()) < epsilon:
				print("Trained", i, "steps")
				break
			
			prev_loss = loss.item()

		self.gp.eval()
		self.likelihood.eval()

		ls = self.base_covar_kernel.lengthscale
		print(f"Lengthscale:{ls}")

	# TODO: Could be improved with fast_pred_var?
	def execute(self, inputs, with_grad=False):
		inputs = torch.tensor(inputs, dtype=torch.double)
		inputs = (inputs-self.train_mean)/self.train_std
		
		if with_grad:
			inputs.requires_grad = True

			pred = self.likelihood(self.gp(inputs))
			mean_pred = pred.mean.sum()

			mean_pred.backward(retain_graph=True)
			grad_mean = inputs.grad


			return (pred.mean.detach().numpy(), pred.variance.detach().numpy(), torch.max(torch.abs(grad_mean)).item())
		else:
			with torch.no_grad():
				f_preds = self.gp(inputs)
				return (f_preds.mean.numpy(), f_preds.variance.numpy())

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
	test = OrdinaryKriging()
	test.train(train_x, train_y)

	test_x = torch.tensor([0.25])
	f_preds = test.gp(test_x)
	print(f_preds.mean)