import os
from abc import ABC, abstractmethod

from simulation import MCP_Manager

class BaseObjectiveFunction(ABC):
	def __init__(self, name, dim):
		self.name = name
		self.dim = dim
		self.failure_probability = -1
		self.mcp_manager = MCP_Manager()
		if not self.mcp_manager.data_exists(self.name):
			print("Monte-Carlo population does not exist, generating...")
			self.mcp_manager.generate_data(self.name, self.data_definition, min(10**6, 10**(dim+3)))

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
		pass

	@abstractmethod
	def logpdf(self, x):
		pass