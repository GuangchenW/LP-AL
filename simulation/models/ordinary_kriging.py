import numpy as np

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.regr import regr_constant

import matplotlib.pyplot as plt

from .model import BaseModel

class OrdinaryKriging(BaseModel):
	def __init__(self, n_dim, regression=None, correlation=None):
		super().__init__()
		self.regression = regression or regr_linear
		self.correlation = correlation or corr_gauss
		theta = np.ones(n_dim)
		thetaL = theta * 1e-4
		thetaU = theta * 20
		self.dacefit = DACE(regr=self.regression, corr=self.correlation, theta=theta, thetaL=thetaL, thetaU=thetaU)

	def train(self, inputs, targets):
		self.dacefit.fit(inputs, targets)

	def execute(self, inputs, with_grad=False):
		result = self.dacefit.predict(inputs, return_mse=True, return_gradient=with_grad)
		if with_grad:
			return result[0].flatten(), result[1].flatten(), result[2]
		else:
			return result[0].flatten(), result[1].flatten()

def test_func(inputs):
	return np.sum(np.sin(inputs*2*np.pi), axis=1)

def test_func2(inputs):
	return inputs[0]**2+inputs[1]**2

if __name__ == "__main__":
	train_x = np.array([[0,0],[0.2,0.4], [0.3, 0.1], [0.5,0.3], [0.17, 0.22], [0.8, 0.6], [0.32,0.45], [2.2,1.8]])
	train_y = np.array([test_func2(x) for x in train_x])
	test_x = np.linspace(0,2,100)[:,None]
	test_x = np.array([[-0.1,-0.5],[0.25,0.25], [0.5,0.5], [1,1], [1.5,1.5], [2.0,2.0]])
	model = OrdinaryKriging(2)
	model.train(train_x,train_y)
	test_y, mse, grad = model.execute(test_x, with_grad=True)
	print(test_y)
	print(mse)
	print(grad)
	exit()
	plt.scatter(test_x, test_y, label="estimate")
	plt.plot(train_x, train_y, label="data")
	plt.legend()
	plt.show()