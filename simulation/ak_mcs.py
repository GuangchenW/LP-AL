import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from gpytorch.kernels import ScaleKernel, RBFKernel

from .utils import ESC, Logger
from models import GPRegression, OrdinaryKriging
from acquisition_functions import U
from evaluators import LP_Batch
from subset_samplers import U_Sampler

class AKMCS:
	def __init__(self, acq_func=None, sampler=None, evaluator=None, max_iter=400, batch_size=1):
		self.acq_func = acq_func if not acq_func == None else U()
		self.sampler = sampler if not sampler == None else U_Sampler(threshold=4)
		#self.sampler.aggressive_mode(True)
		if evaluator:
			self.evaluator = evaluator
			self.evaluator.acq_func = self.acq_func
		else:
		 	self.evaluator = LP_Batch(acq_func=self.acq_func)
		self.stopper = ESC(epsilon_thr=0.01)
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.name = self.evaluator.name+"_"+self.acq_func.name

	# TODO2: implement easier to use verbose settings
	def initialize_input(self, 
		obj_func, 
		sample_size=None, 
		bootstrap_inputs=None, 
		bootstrap_outputs=None, 
		seed=1, 
		debug=False, 
		silent=True):
		"""
		Reset and initialize the objective function and input data. 
		:param obj_func: The objective function object.
		:param sample_size: Number of points from the `input_space` used for kriging estimation.
		:param bootstrap_inputs: Initial input samples for priming the Kriging model. If none, a dozen(s) of samples will be chosen randomly.
		:param bootstrap_outputs: The outputs associated with `bootstrap_inputs`.
		:param seed: Random seed for the algorithm. Deafults to 1.
		"""
		self.obj_func = obj_func
		self.model = OrdinaryKriging(n_dim=self.obj_func.ndim)
		self.input_space = obj_func.load_data()
		self.sample_size = sample_size if sample_size else 10**4
		# If no initial inputs provided, use a numer of random samples equal to 
		# the smallest multiple of 12 larger than the dimension of the system.
		self.num_init = bootstrap_inputs.shape[0] if bootstrap_inputs else 12 * (obj_func.ndim // 12 + 1)
		self.sampler.obj_func = obj_func

		self.prob_history = []

		np.random.seed(seed)
		np.random.shuffle(self.input_space)
		self.kriging_sample = self.input_space[:self.sample_size]

		self.training_inputs = self.obj_func.normalize_data(bootstrap_inputs) if bootstrap_inputs else self.kriging_sample[:self.num_init]
		self.training_outputs = np.copy(bootstrap_outputs) if bootstrap_outputs else np.array([self.obj_func.evaluate(x, True) for x in self.training_inputs])
		self.sample_history = []

		# Logging setup
		log_file_name = "%s_%s_init%d_batch%d.txt" % (obj_func.name, self.acq_func.name, self.num_init, self.batch_size)
		self.logger = Logger(log_file_name, silent=silent, active=debug)
		self.silent = silent

	def kriging_estimate(self, do_mcs=False):
		for i in range(self.max_iter):
			if self.silent:
				print("%s | %s | iter : %d" % (self.acq_func.name, self.obj_func.name, i))
			if self.kriging_step(i):
				break
		
		result = self.compute_failure_probability(do_mcs=do_mcs)
		result["iter"] = i

		print(repr(self.prob_history))
		return result

	def check_convergence(self, mean, variance):
		"""
		:return: `True` if the relative error has converged, `False` otherwise.
		"""
		# Compute estimated relative error epsilon_max
		epsilon_max, should_stop = self.stopper(mean, variance)
		self.logger.log("Epsilon max : %.6g" % epsilon_max)
		return should_stop

	def check_coefficient_variation(self, mean, threshold=0.05):
		"""
		:return: `True` if coefficient of variation is less or equal to threshold, `False` otherwise.
		"""
		# Compute coefficient of variation
		num_failures = np.sum(mean < 0)
		est_prob_failure = num_failures / self.sample_size
		cv = np.sqrt((1-est_prob_failure)/(est_prob_failure*self.sample_size))

		# Check if coefficient of variation is acceptable.
		# If not, expand sample pool.
		if cv > threshold:
			if self.sample_size >= 10**5:
				# Good enough, don't want to spend more time.
				return True
			self.sample_size += 10**4
			self.kriging_sample = self.input_space[:self.sample_size]
			return False
		else:
			return True

	def kriging_step(self, iter_count):
		"""
		One iteration of the LP-AL algorithm.
		1) Fit Kriging model
		2) Check convergence/coefficient of variation
		3) Update adaptive sample pool
		4) Compute L and acquire batch
		5) Update training set
		"""
		# Construct Kriging model
		self.model.train(self.training_inputs, self.training_outputs)

		while True:
			# Obtain mean, variance and expected gradient for all samples
			mean, variance, grad = self.model.execute(self.kriging_sample, with_grad=True)
			
			self.prob_history.append(np.sum(mean<0)/len(mean))

			has_converged = self.check_convergence(mean, variance)

			if has_converged:
				# If coefficient of variation is sufficiently low, tell process to finish.
				# Otherwise re-evaluated convergence criterion with the expanded sample pool.
				if self.check_coefficient_variation(mean):
					return True
			else:
				# Not yet converged, carry on with batch acquisition.
				break

		# Sample critical region
		for i in range(16):
			subset_pop, subset_mean, subset_var = self.sampler.sample(
				self.kriging_sample, 
				self.training_inputs, 
				self.training_outputs,
				mean,
				variance)

			if subset_pop.shape[0] >= self.batch_size:
				self.sample_history.append(subset_pop)
				break
			else:
				self.sampler.threshold += 2
				self.logger.log("Stopping condition not met, increasing sampling threshold to %d." % self.sampler.threshold)
		
		if subset_pop.shape[0] <= 0:
			self.logger.log("Cannot further expand critical region. Stopping early.")
			return False

		self.logger.log("Subset size: %d" % len(subset_pop))

		# Feed the reduced candidate sample pool to evaluator and get batch
		self.evaluator.set_grad(grad)
		batch = self.evaluator.obtain_batch(
			subset_pop, 
			subset_mean, 
			subset_var, 
			self.training_inputs, 
			self.training_outputs, 
			self.batch_size)

		# Update training set with batch
		for candidate in batch:
			self.training_inputs = np.append(self.training_inputs, [candidate["next"]], axis=0)
			self.training_outputs = np.append(
				self.training_outputs, 
				self.obj_func.evaluate(candidate["next"], True))

		# Record selections
		self.logger.log_batch(iter_count, batch)

		return False

	def compute_failure_probability(self, do_mcs):
		# Compute coefficient of variation
		N_MCS = self.input_space.shape[0]
		mean, var = self.model.execute(self.input_space)
		num_failures = np.sum(mean < 0)
		est_prob_failure = num_failures / N_MCS
		cov_fail = np.sqrt((1-est_prob_failure)/(est_prob_failure*N_MCS))
		
		# TODO: Clean this up
		MCS_prob_failure = float("nan")
		if do_mcs:
			MCS_N_f = 0
			for i in range(N_MCS):
				if self.obj_func.evaluate(self.input_space[i], True) < 0:
					MCS_N_f += 1
				print("MCS progress [%d/1000000]\r" % i, end="")
			MCS_prob_failure = MCS_N_f/N_MCS
		else:
			if self.obj_func.failure_probability > 0:
				MCS_prob_failure = self.obj_func.failure_probability

		self.logger.log(f"True probability of failure: {MCS_prob_failure:.6g}")
		self.logger.log(f"Estimated probability of failure: {est_prob_failure:.6g}")
		self.logger.log(f"COV of probability of failure: {cov_fail:.6g}")
		self.logger.clean_up()
		return {
			"system": self.obj_func.name,
			"Pf":  MCS_prob_failure,
			"name": self.acq_func.name,
			"Pfe": est_prob_failure,
			"COV": cov_fail,
			"re": abs(MCS_prob_failure-est_prob_failure)/MCS_prob_failure
		}

	def visualize(self, mk_sample_pool_anim=False, save_visual=False, filename=["limit_state_plot.png","sample_selections.gif"] ):
		"""
		Visualize the Kriging model and estimated limit state.
		Only works for 2D systems.

		:param mk_sample_pool_anim: Whether to make an animation of adaptive sample pool changing over time. (Default: `False`)
		:param save_visual: Whether to save the visualizations. (Default: `False`)
		:param filename: Files to save to. (Deafult: ``["plot.png","sample_pool.gif"]``)
		"""
		if not self.training_inputs.shape[1] == 2:
			print("Visualization only available for systems with 2 variables!")
			return

		matplotlib.rcParams["mathtext.fontset"]="cm"
		############################################################
		# Animate the adaptive sample pool evolves over time
		if mk_sample_pool_anim and len(self.sample_history) > 0:
			fig, ax = plt.subplots()
			artists = []
			for i in range(len(self.sample_history)):
				samples = np.array(self.sample_history[i]).T
				sample_plot = ax.scatter(samples[0], samples[1], c="blue", s=2)
				doe_range_l = self.num_init+i*self.batch_size
				doe_range_u = self.num_init+(i+1)*self.batch_size
				selection = self.training_inputs[doe_range_l:doe_range_u].T
				selection_plot = ax.scatter(selection[0], selection[1], marker="*", c="red", s=10)
				txt = ax.text(0.05,0.05, str(i), ha="right", va="bottom", transform=fig.transFigure)
				artists.append([sample_plot, selection_plot, txt])
			ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=200)
			plt.show()
		############################################################
		# Visualization
		# Density of grid for visualization
		N_GRID = 400
		grid_x = np.linspace(-4.2, 4.2, N_GRID)
		grid_y = np.linspace(-4.2, 4.2, N_GRID)
		xpts, ypts = np.meshgrid(grid_x, grid_y)
		pts = np.dstack((xpts.ravel(), ypts.ravel()))
		z, ss = self.model.execute(pts[0])
		z = z.reshape((N_GRID,N_GRID))

		fig, ax = plt.subplots()
		contours = ax.contourf(grid_x, grid_y, z, levels=100, cmap='YlGn_r')
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(contours, label="value", cax=cax)
		ax.set_aspect(0.75)
		cax.set_aspect(15)
		ax.set_xlabel(r"$x_1$", fontsize=12)
		ax.set_ylabel(r"$x_2$", fontsize=12)
		#plt.title('Kriging Interpolation')

		# Query G on the grid
		G_values = np.zeros((N_GRID, N_GRID))
		for i in range(len(grid_x)):
			for j in range(len(grid_y)):
				G_values[i,j] = self.obj_func.evaluate([grid_x[i], grid_y[j]])

		# Actual limit state i.e. f(x, y)=0
		contours = ax.contour(xpts, ypts, G_values, levels=[0], colors='blue')

		# Estimated limit state by the kriging model
		contours = ax.contour(grid_x, grid_y, z, levels=[0], colors="red", linewidths=2, linestyles='dashed')

		# Plot the initial training samples
		ax.scatter(self.training_inputs[:self.num_init, 0], self.training_inputs[:self.num_init, 1], marker="v", s=5, c="magenta", label=r"$\mathcal{B}_0$")
		# Plot the samples added
		ax.scatter(self.training_inputs[self.num_init:, 0], self.training_inputs[self.num_init:, 1], marker="o", s=5, c="black", label="Samples added")

		# Legend for the acual and estimated limit state
		handles, _ = ax.get_legend_handles_labels()
		est_limit_state = Line2D([0], [0], label=r"$\hat{f}(\mathbf{x})=0$", color="red", linestyle='dashed')
		actual_limit_state = Line2D([0], [0], label=r"$f(\mathbf{x})=0$", color="blue")
		handles.extend([est_limit_state,actual_limit_state])
		ax.legend(handles=handles, loc=1, prop={"size":10})

		plt.tight_layout()
		if save_visual:
			plt.savefig(filename[0], dpi=300)
			if mk_sample_pool_anim:
				ani.save(filename[1])
			print("Visualization saved!")
		else:
			plt.show()