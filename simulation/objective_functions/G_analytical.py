from .objective_function import BaseObjectiveFunction


class AnalyticalFunction(BaseObjectiveFunction):
	def __init__(self, name, ndim, variable_definition, variable_logpdf, limit_state_function, failure_probability=None):
		"""
		:param name: Name of this function.
		:param ndim: Number of variables.
		:param variable_definition: A function that takes no argument. Returns a randomly generated input sample as numpy array.
		:param variable_logpdf: A function that takes an input sample as argument. Returns the log joint pdf of the input sample.
		:param limit_state_function: A function that takes an input sample as argument. Returns the evaluation of the limit-state function.
		:param failure_probability: Precomputed true probability of failure. If ``None`` or negative, MCS will be used. (Default: ``None``) 
		"""
		self.variable_definition = variable_definition
		self.variable_logpdf = variable_logpdf
		self.limit_state_function = limit_state_function
		super().__init__(name=name, ndim=ndim, failure_probability=failure_probability)

	def _evaluate(self, x):
		return self.limit_state_function(x)

	def variable_definition(self):
		return self.variable_definition()

	def logpdf(self, x):
		return self.variable_logpdf(self.denormalize_data(x))