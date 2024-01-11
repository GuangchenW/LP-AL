from abc import ABC, abstractmethod

class BaseAcquisitionFunction(ABC):
	def __init__(self, name=None, device="cpu", logger=None):
		self.name = name
		self.device = device
		self.logger = logger

	@abstractmethod
	def acquire(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
	):
		"""
		Returns acq(x) for all x in `subset_points`.
		:param subset_points: The subset of the monte-carlo population for this acquisition.
		:param mean: the estimated mean of `subset_points`.
		:param variance: the estimated variance of `subset_points`.
		:param doe_input: All inputs evaluated so far, including the initial DOE.
		:param doe_response: Evaluations for `doe_input`.
		"""