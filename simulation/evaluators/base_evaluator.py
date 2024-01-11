from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
	def __init__(self, acq_func, logger=None):
		self.acq_func = acq_func
		self.logger = logger

	@abstractmethod
	def obtain_batch(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
		n_points
	):
		"""
		:param subset_points: The subset of the monte-carlo population for this acquisition.
		:param mean: the estimated mean of `subset_points`.
		:param variance: the estimated variance of `subset_points`.
		:param doe_input: All inputs evaluated so far, including the initial DOE.
		:param doe_response: Evaluations for `doe_input`.
		:param n_points: Number of points to acquire. For single acquisition, this must be 1.
		"""