from abc import ABC, abstractmethod

class BaseModel(ABC):
	def __init__(self):
		pass

	@abstractmethod
	def train(self, inputs, targets, device="cpu"):
		"""
		Train the model using the supplied inputs and targets.
		:param inputs: The training points.
		:param targets: The observations at the training points, assumed to be noiseless.
		:param device: The device used for training. (Currently only supports CPU)
		"""
	
	@abstractmethod
	def execute(self, inputs, with_grad=False):
		"""
		Returns the estimations at the given input points.
		:param inputs: The points with unknown observations.
		:param with_grad: Whether to also return the gradient of the estimated mean.
		"""