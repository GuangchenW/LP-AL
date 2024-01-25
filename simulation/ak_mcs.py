import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from gpytorch.kernels import ScaleKernel, RBFKernel

from .logger import Logger
from .utils import ESC
from models import OrdinaryKriging
from acquisition_functions import ULP
from evaluators import LP_Batch
from subset_samplers import U_Sampler

class AKMCS:
	def __init__(self, model=None, acq_func=None, sampler=None, evaluator=None, max_iter=100, batch_size=1):
		#self.model = model if not model == None else OrdinaryKriging(covar_kernel = ScaleKernel(RBFKernel(ard_num_dims=n_dim)))
		self.acq_func = acq_func if not acq_func == None else ULP()
		self.sampler = sampler if not sampler == None else U_Sampler(threshold=2)
		self.evaluator = evaluator if not evaluator == None else LP_Batch(acq_func=self.acq_func)
		self.stopper = ESC(epsilon_thr=0.01)
		self.max_iter = max_iter
		self.batch_size = batch_size

	def initialize_input(self, obj_func, sample_size=None, num_init=12, lengthscale=None, random=False):
		"""
		Reset and initialize the objective function and input data. 
		:param input_space: The monte-carlo population.
		:param obj_func: The objective function.
		:param sample_size: Number of points from the `input_space` used for kriging estimation.
		:param num_init: How many points to evaluate for the initial DOE.
		:param random: If true, the initial DOE and samples for kriging will be taken randomly.
		"""
		self.obj_func = obj_func
		self.input_space = obj_func.load_data()
		if sample_size == None:
			self.kriging_sample = self.input_space
		else:
			sample_size = min(self.input_space.shape[0], sample_size)
			self.kriging_sample = self.input_space[:sample_size] if not random else np.random.Generator.choice(self.input_space, sample_size, replace=False)
		self.num_init = num_init
		self.doe_input = self.kriging_sample[:num_init] if not random else np.random.Generator.choice(self.kriging_sample, num_init, replace=False)
		self.doe_response = np.array([self.obj_func.evaluate(x) for x in self.doe_input])
		print(self.doe_response)
		self.model = OrdinaryKriging(n_dim=self.obj_func.dim, init_lengthscale=lengthscale)
		self.sample_history = []

		log_file_name = "%s_%s_init%d_batch%d.txt" % (obj_func.name, self.acq_func.name, num_init, self.batch_size)
		self.logger = Logger(log_file_name, silence=True)

	def kriging_estimate(self):
		for i in range(self.max_iter):
			if not self.kriging_step(i):
				break

	def kriging_step(self, iter_count):
		# STEP3 Compute Kriging model
		self.model.train(self.doe_input, self.doe_response)
		# Acquire all estimations
		mean, variance, max_grad = self.model.execute(self.kriging_sample, with_grad=True)

		# Compute stopping criterion
		epsilon_max, should_stop = self.stopper(mean, variance)
		self.logger.log("Epsilon max : %.6g" % epsilon_max)
		if should_stop:
			return False

		# Sample critical region
		for i in range(10):
			subset_pop, subset_mean, subset_var = self.sampler.sample(
				self.kriging_sample, 
				self.doe_input, 
				self.doe_response,
				mean,
				variance)

			if subset_pop.shape[0] >= self.batch_size:
				self.sample_history.append(subset_pop)
				break
			else:
				self.sampler.threshold += 1
				self.logger.log("Stopping condition not met, increasing sampling threshold to %d." % self.sampler.threshold)
		
		if subset_pop.shape[0] <= 0:
			self.logger.log("Cannot further exapnd critical region. Stopping early.")
			return False

		self.logger.log("Subset size: %d" % len(subset_pop))
		self.logger.log("Max Expected Gradient: %.4g" % max_grad)

		# STEP5 Compute learning function on the population and identify best point
		self.evaluator.set_L(max_grad)
		batch = self.evaluator.obtain_batch(
			subset_pop, 
			subset_mean, 
			subset_var, 
			self.doe_input, 
			self.doe_response, 
			self.batch_size)

		# STEP6 Update doe with batch
		for candidate in batch:
			self.doe_input = np.append(self.doe_input, [candidate["next"]], axis=0)
			self.doe_response = np.append(self.doe_response, self.obj_func.evaluate(candidate["next"]))

		self.logger.log_batch(iter_count, batch)

		return True

	#TODO
	def tail(self):
		# STEP8: Compute coefficient of variation of the probability of failure
		N_MC = self.input_space.shape[0]
		z, ss = self.model.execute(self.input_space)
		num_negative_predictions = np.sum(z < 0)
		P_f = num_negative_predictions / N_MC
		cov_fail = np.sqrt((1-P_f)/(P_f*N_MC))

		N_true_f = 0
		for i in range(N_MC):
			if self.obj_func.evaluate(self.input_space[i]) < 0:
				N_true_f += 1

		self.logger.log(f"True probability of failure: {N_true_f/N_MC:.3g}")
		self.logger.log(f"Estimated probability of failure: {P_f:.3g}")
		self.logger.log(f"COV of probability of failure: {cov_fail:.3g}")
		self.logger.clean_up()
		#return

		if not self.doe_input.shape[1] == 2:
			return
		############################################################
		# subset sample evolution
		if len(self.sample_history) > 0:
			fig, ax = plt.subplots()
			artists = []
			for i in range(len(self.sample_history)):
				samples = np.array(self.sample_history[i]).T
				sample_plot = ax.scatter(samples[0],samples[1], c="blue", s=2)
				doe_range_l = self.num_init+i*self.batch_size
				doe_range_u = self.num_init+(i+1)*self.batch_size
				selection = self.doe_input[doe_range_l:doe_range_u].T
				selection_plot = ax.scatter(selection[0], selection[1],c="red",s=4)
				txt = ax.text(0.05,0.05, str(i), ha="right", va="bottom", transform=fig.transFigure)
				artists.append([sample_plot, selection_plot, txt])
			ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=8000/len(self.sample_history))
			plt.show()


		############################################################
		# Visualization
		# Density of grid for visualization
		N_GRID = 400
		grid_x = np.linspace(-5, 5, N_GRID)
		grid_y = np.linspace(-5, 5, N_GRID)
		xpts, ypts = np.meshgrid(grid_x, grid_y)
		pts = np.dstack((xpts.ravel(), ypts.ravel()))
		z, ss = self.model.execute(pts)
		z = z.reshape((N_GRID,N_GRID))

		plt.figure()
		contours = plt.contourf(grid_x, grid_y, z, levels=100, cmap='jet')

		plt.colorbar(label='Value')
		plt.xlabel('X1-coordinate')
		plt.ylabel('X2-coordinate')
		plt.title('Kriging Interpolation')

		# Level 0, the estimate of the limit state by the kriging model
		contours = plt.contour(grid_x, grid_y, z, levels=[0], colors='r', linewidths=2)

		# Plot the points queried
		plt.scatter(self.doe_input[:, 0], self.doe_input[:, 1], s=2, c='black', label='Data')
		# Label the points queried with their actual value
		#for x1, x2, h in DOE:
		#    plt.text(x1, x2, f'{h:.2f}', fontsize=8, color='white', ha='center', va='center')

		# Kriging model contour
		plt.contour(grid_x, grid_y, z, colors='white', linewidths=1, linestyles='dashed', alpha=0.5)

		# Color bar and legends
		plt.colorbar(label='Value')
		plt.legend()

		# Mesh
		x1_grid, x2_grid = np.meshgrid(grid_x, grid_y)

		# Query G on the grid
		G_values = np.zeros((N_GRID, N_GRID))
		for i in range(len(grid_x)):
			for j in range(len(grid_y)):
				G_values[i,j] = self.obj_func.evaluate([grid_x[i], grid_y[j]])

		# Actual limit state i.e. G(x1, x2)=0
		contours = plt.contour(x1_grid, x2_grid, G_values, levels=[0], colors='b', linestyles='dashed')

		plt.show()

		ani.save("G_Ras.gif")