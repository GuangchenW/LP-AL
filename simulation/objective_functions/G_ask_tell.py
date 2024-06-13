from .objective_function import BaseObjectiveFunction

import os
import numpy as np
from scipy.stats import norm

class AskTellFunction(BaseObjectiveFunction):
	def __init__(self, name, ndim, data_definition, failure_probability, data_logpdf=None):
		"""
		The Ask-Tell interface will not perform MCS on the limit-state function 
		and thus requires a precomputed probability of failure for comparison. If a comparison between 
		the estimation and the truth is undesirable, `do_mcs` should be False when calling `kriging_estimate`.
		"""
		self.data_definition = data_definition
		self.data_logpdf = data_logpdf
		super().__init__(name=name, ndim=ndim, failure_probability=failure_probability)

	def _evaluate(self, x):
		prompt = "Enter output for %s: " % str(x)

		while True:
			try:
				response = float(input(prompt))
				break
			except ValueError as ve:
				print("Please enter a number.")

		return response

	def data_definition(self):
		x = self.data_definition()
		print(x)
		return x

	def logpdf(self, x):
		return self.data_logpdf(self.denormalize_data(x))