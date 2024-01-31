import sys
import os
import math
import numpy as np
import scipy.stats as st

class ESC:
	def __init__(self, confidence_level=0.05, epsilon_thr=0.05):
		self.confidence_level = confidence_level
		self.lambda_confidence = st.norm.ppf(confidence_level/2) # Confidence interval normal
		self.epsilon_thr = epsilon_thr

	def __call__(self, mean, variance):
		epsilon_max = self.compute_epsilon_max(mean, variance)
		should_stop = epsilon_max <= self.epsilon_thr
		return epsilon_max, should_stop

	def compute_epsilon_max(self, mean, variance):
		N_f = np.sum(mean<=0)
		if N_f == 0:
			return math.inf
		mu_Ss, mu_Sf, var_Ss, var_Sf = self.compute_wrong_sign_stats(mean, variance)
		false_safe_upper = self.compute_false_safe_upper_bound(mu_Ss, var_Ss)
		false_fail_upper = self.compute_false_fail_upper_bound(mu_Sf, var_Sf)
		prob1 = abs(N_f/(N_f-false_fail_upper)-1)
		prob2 = abs(N_f/(N_f+false_safe_upper)-1)
		return max(prob1, prob2)

	def compute_false_safe_upper_bound(self, mean_false_safe, var_false_safe):
		return mean_false_safe + self.lambda_confidence * np.sqrt(var_false_safe)

	def compute_false_fail_upper_bound(self, mean_false_fail, var_false_fail):
		return st.poisson.ppf(1-self.confidence_level/2, mean_false_fail)

	def compute_wrong_sign_stats(self, mean, variance):
		"""
		Compute the expected number and variance of wrong sign estimation for both false positives and negatives.
		However, these cannot be used as is because the number of false positives and negatives
		are still randomly distributed.
		"""
		mask_uncertain = ~(variance == 0)
		mean = mean[mask_uncertain]
		variance = variance[mask_uncertain]

		mask_safe = mean >= 0
		mask_fail = ~mask_safe

		z = -abs(mean/np.sqrt(variance))
		prob_wrong_sign = st.norm.cdf(z)

		prob_wrong_sign_safe = prob_wrong_sign[mask_safe]
		prob_wrong_sign_fail = prob_wrong_sign[mask_fail]

		mu_Ss = np.sum(prob_wrong_sign_safe) ##Expected number of false safe
		mu_Sf = np.sum(prob_wrong_sign_fail) ##Expected number of false fail

		var_Ss = np.sum(prob_wrong_sign_safe*(1-prob_wrong_sign_safe))
		var_Sf = np.sum(prob_wrong_sign_fail*(1-prob_wrong_sign_fail))

		return mu_Ss, mu_Sf, var_Ss, var_Sf

class Logger:
	def __init__(self, out_target=None, silent=False):
		directory = os.path.dirname(os.path.realpath(__file__))
		target_path = directory+"/test_logs/"
		if not os.path.exists(target_path):
			os.makedirs(target_path)
		self.out_target = open(target_path+out_target, "w") if out_target else sys.stdout
		self.silent = silent

	def log(self, content):
		self.out_target.write(content+"\n")
		if not self.silent and self.out_target is not sys.stdout:
			print(content)

	def log_batch(self, iteration, batch):
		self.log("**"*25)
		self.log("Iteration [%i] | Batch size [%i]" % (iteration, len(batch)))
		for candidate in batch:
			self.log("--"*25)
			self.log("Selected" + str(candidate["next"]))
			self.log("SCORE : %.3f | MEAN : %.4g | VARIANCE : %.4g" % (
				candidate["utility"],
				candidate["mean"],
				candidate["variance"]))
		self.log("**"*25)

	def clean_up(self):
		if self.out_target is not sys.stdout:
			self.out_target.close()
