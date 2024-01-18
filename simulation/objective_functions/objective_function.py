import os
from abc import ABC, abstractmethod

from simulation import MCP_Manager

class BaseObjectiveFunction(ABC):
	def __init__(self, name, dim):
		self.name = name
		self.dim = dim
		self.mcp_manager = MCP_Manager()
		if not self.mcp_manager.data_exists(self.name):
			print("Monte-Carlo population does not exist, generating...")
			self.mcp_manager.generate_data(self.name, self.data_definition, min(10**6, 10**(dim+2)))

	@abstractmethod
	def evaluate(self, x):
		pass

	def load_data(self):
		return self.mcp_manager.load_data(self.name)

	@abstractmethod
	def data_definition(self):
		pass