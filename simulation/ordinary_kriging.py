import math
import torch
import gpytorch

from gpytorch.models import ExactGP
from gpytorch.means import Mean, ConstantMean, ZeroMean
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
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

class OrdinaryKriging:
	def __init__(self, likelihood=None, mean_func=None, covar_kernel=None):

		self.mean_func = mean_func or ConstantMean()

		if covar_kernel is None:
			self.covar_kernel = ScaleKernel(MaternKernel())

		self.likelihood = likelihood or GaussianLikelihood()
		self.gp = None

	def train(self, inputs, labels, device="cpu"):
		inputs = torch.tensor(inputs)
		labels = torch.tensor(labels)

		if self.gp == None:
			self.gp = _ExactGP(inputs, labels, self.likelihood, self.mean_func, self.covar_kernel)
			self.gp.to(device)
		self.gp.train()
		self.likelihood.train()

		optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)

		mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)

		max_iter = 1000
		epsilon = 0.001
		prev_loss = 100
		for i in range(max_iter):
			optimizer.zero_grad()
			output = self.gp(inputs)
			loss = -mll(output, labels)
			loss.backward()
			"""
			print('Iter %d/%d - Loss: %.3f		lengthscale: %.3f	noise: %.3f' % (
				i+1, max_iter, loss.item(),
				self.gp.covar_kernel.base_kernel.lengthscale.item(),
				self.gp.likelihood.noise.item()
				))
			"""
			if abs(prev_loss-loss.item()) < epsilon:
				print("Trained", i, "steps")
				break
			else:
				prev_loss = loss.item()

			optimizer.step()
		self.gp.eval()
		self.likelihood.eval()

	# TODO: Could be improved with fast pred var?
	def execute(self, inputs):

		with torch.no_grad():
			inputs = torch.tensor(inputs)
			f_preds = self.gp(inputs)
			return (f_preds.mean.numpy()[0], f_preds.variance.numpy()[0])

if __name__ == "__main__":
	train_x = torch.linspace(0,1,100)
	train_y = torch.sin(train_x*2*math.pi+torch.randn(train_x.size())*math.sqrt(0.04))
	test = OrdinaryKriging()
	test.train(train_x, train_y)

	test_x = torch.tensor([0.25])
	f_preds = test.gp(test_x)
	print(f_preds.mean)