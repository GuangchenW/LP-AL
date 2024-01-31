import math
import torch
import numpy as np
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

train_x = torch.linspace(0,1,100)
train_y = torch.sin(train_x*2*math.pi+torch.randn(train_x.size())*math.sqrt(0.04))
train_x = train_x.numpy().reshape(-1,1)
train_y = train_y.numpy()

rng = np.random.RandomState(1)
train_idx = rng.choice(np.arange(train_y.size), size=6, replace=False)

kernel = 1*RBF(1.0)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gpr.fit(train_x[train_idx],train_y[train_idx])
print(gpr.kernel_)
mean, std = gpr.predict(train_x, return_std=True)
print(mean, std)