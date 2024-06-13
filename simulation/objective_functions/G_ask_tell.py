from .objective_function import BaseObjectiveFunction

class AskTellFunction(BaseObjectiveFunction):
	def __init__(self, name, ndim, variable_definition, variable_logpdf, failure_probability):
		"""
		The Ask-Tell interface will not perform MCS on the limit-state function 
		and thus requires a precomputed probability of failure for comparison. If a comparison between 
		the estimation and the truth is undesirable, ``do_mcs`` should be False when calling ``kriging_estimate``.

		:param name: Name of this function.
		:param ndim: Number of variables.
		:param variable_definition: A function that takes no argument. Returns a randomly generated input sample as numpy array.
		:param variable_logpdf: A function that takes an input sample as argument. Returns the log joint pdf of the input sample.
		:param failure_probability: Precomputed true probability of failure. If ``None`` or negative, MCS will be used. (Default: ``None``) 
		"""
		self.variable_definition = variable_definition
		self.variable_logpdf = variable_logpdf
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

	def variable_definition(self):
		return self.variable_definition()

	def logpdf(self, x):
		return self.variable_logpdf(self.denormalize_data(x))