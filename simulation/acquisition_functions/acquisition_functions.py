from abc import ABC, abstractmethod
from simulation.utility_functions import U, ULP, NEFF

class BaseAcquisitionFunction(ABC):
	def __init__(self, utility_func, device="cpu", logger=None):
		self.name = utility_func
		match utility_func:
			case "U":
				self.utility_func = U
			case "ULP":
				self.utility_func = ULP
			case "NEFF":
				self.utility_func = NEFF
		self.device = device
		self.logger = logger

	@abstractmethod
	def acquire(
		self,
		input_population,
		doe_input,
		doe_response,
		mean,
		variance,
		n_points
	):
		"""
		:param input_population: the subset of the monte-carlo population used for this current acquisition
		:param doe_input: all inputs acquired so far, including initial doe
		:param doe_response: outputs observed for `doe_input`
		:param mean: the estimated mean of `input_population`
		:param varaince: the estimated variance of `input_population`
		:param n_points: how many points to acquire. For non-batch AFs, this must be 1
		"""

class AcquisitionFunction(BaseAcquisitionFunction, ABC):
    """All sequential acquisition classes should subclass this."""

class BatchAcquisitionFunction(BaseAcquisitionFunction, ABC):
    """All batch acquisition classes should subclass this."""