import os
from abc import ABC, abstractmethod

from simulation import MCP_Manager

class BaseObjectiveFunction(ABC):
	def __init__(self, name, ndim, failure_probability=None):
		self.name = name
		self.ndim = ndim
		self.failure_probability = failure_probability if failure_probability else -1
		self.mcp_manager = MCP_Manager()
		if not self.mcp_manager.data_exists(self.name):
			print("Monte-Carlo population does not exist, generating...")
			self.mcp_manager.generate_data(self.name, self.data_definition, min(10**6, 10**(ndim+3)))

	def evaluate(self, x, denormalize=False):
		if denormalize:
			x = self.denormalize_data(x)
		return self._evaluate(x)

	@abstractmethod
	def _evaluate(self, x):
		pass

	def load_data(self):
		self.actual_data = self.mcp_manager.load_data(self.name)
		self.mean = self.actual_data.mean(axis=0)
		self.std = self.actual_data.std(axis=0)
		return self.normalize_data(self.actual_data)

	def normalize_data(self, data):
		return (data - self.mean)/self.std

	def denormalize_data(self, data):
		return data*self.std+self.mean

	@abstractmethod
	def data_definition(self):
		"""
		Generate a data sample based on the system variable distributions.
		:return: numpy array.
		"""
		pass

	@abstractmethod
	def logpdf(self, x):
		"""
		Returns the probability density at x. May need to be exponetiated at some point.
		At power of ~-755 python precision insufficient and just gives 0, but such a small probability density shouldn't matter anyway.
		:return: The log pdf of x based on data definition. 
		"""
		pass

	def estimate_failure_prob(self, free=False, n=10**7):
		"""
		Used for standalone MCS of the probability of failure for this objective function.
		:param free: If True, samples will be generated on the fly. Otherwise, use saved samples.
		:param n: Number of samples to use. Default 1e+7.
		"""
		n_fail = 0
		if free:
			for i in range(n):
				if self._evaluate(self.data_definition()) < 0:
					n_fail += 1
				print("MCS progress [%d/%d]\r" % (i,n), end="")
			print(n_fail/n)
		else:
			data = self.load_data()
			n = len(data)
			for i,d in enumerate(data):
				if self.evaluate(d, True) < 0:
					n_fail += 1
				print("MCS progress [%d/%d]\r" % (i,n), end="")
			print(n_fail/n)