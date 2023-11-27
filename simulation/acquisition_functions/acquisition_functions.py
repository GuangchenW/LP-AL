from abc import ABC, abstractmethod

class BaseAcquisitionFunction(ABC):
	def __init__(self, name, device="cpu", logger=None):
		self.name = name
		self.device = device
		self.logger = logger

	@abstractmethod
	def acquire(
		self,
		model,
		input_population,
		doe_input,
		doe_response,
		n_points
	):
		"""
		:param model: the model used to approximate the system
		:param input_population: the subset of the monte-carlo population used for this current acquisition
		:param doe_input: all inputs acquired so far, including initial doe
		:param doe_response: outputs observed for `doe_input`
		:param n_points: how many points to acquire. For non-batch AFs, this must be 1
		"""

class AcquisitionFunction(BaseAcquisitionFunction, ABC):
    """All sequential acquisition classes should subclass this."""

class BatchAcquisitionFunction(BaseAcquisitionFunction, ABC):
    """All batch acquisition classes should subclass this."""