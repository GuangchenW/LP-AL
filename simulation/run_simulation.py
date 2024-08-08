import os
import datetime
import time

import torch
import numpy as np
import tensorflow as tf
from wakepy import keep

from simulation import AKMCS
from objective_functions import G_Simple, G_4B, G_Ras, G_Oscillator, G_Tube, G_FEM, G_Stiffener, AskTellFunction, AnalyticalFunction
from acquisition_functions import U, EFF, ERF, H, LIF, VAR
from evaluators import LP_Batch, NLP_Batch, KMeans_Batch, KB_Batch
from subset_samplers import U_Sampler, ImportanceSampler, NaiveSampler

def get_test_takers(batch_size, obj_func):
	takers = []
	takers.append(AKMCS(acq_func=U(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=EFF(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=ERF(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=H(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=LIF(obj_func), batch_size=batch_size))
	if batch_size > 1:
		takers.append(AKMCS(acq_func=U(), sampler=U_Sampler(), evaluator=KB_Batch(acq_func=None), batch_size=batch_size))
		takers.append(AKMCS(acq_func=U(), sampler=U_Sampler(), evaluator=KMeans_Batch(acq_func=None), batch_size=batch_size))
	return takers

def get_test_suite():
	#return [G_4B(), G_Ras(), G_Oscillator(), G_Tube()]
	return [G_FEM()]

def run_test_single():
	"""
	data_def = lambda : [np.random.normal(0, 1), np.random.normal(0, 1)]
	def ls_func(x):
		x1,x2 = x
		b1 = 3 + 0.1*(x1-x2)**2 - (x1+x2)/np.sqrt(2)
		b2 = 3 + 0.1*(x1-x2)**2 + (x1+x2)/np.sqrt(2)
		b3 = (x1-x2) + 7/np.sqrt(2)
		b4 = (x2-x1) + 7/np.sqrt(2)
		return np.min([b1, b2, b3, b4])
	def data_logpdf(x):
		x1, x2 = x
		prob = norm.logpdf(x1, 0, 1)
		prob += norm.logpdf(x2, 0, 1)
		return prob
	"""
	#test = AnalyticalFunction(name="TestAskTell", ndim=2, variable_definition=data_def, variable_logpdf=data_logpdf, limit_state_function=ls_func)
	test = G_Stiffener()

	#taker = AKMCS(acq_func=U(), sampler=U_Sampler(), evaluator=KB_Batch(acq_func=None), batch_size=4)
	#taker = AKMCS(acq_func=U(), sampler=U_Sampler(), evaluator=KMedoid_Batch(acq_func=None), batch_size=1)
	taker = AKMCS(acq_func=U(), sampler=U_Sampler(), batch_size=2)
	taker.initialize_input(test, sample_size=10**4, seed=10, silent=False, debug=True) #46?
	result = taker.kriging_estimate(do_mcs=False)
	print(result)
	taker.visualize(mk_sample_pool_anim=True, save_visual=True)

def run_test_suite(idx):
	tests = get_test_suite()
	for i in [1,4,8,12]:
		with open("experiement%d.txt" % idx, "a") as file:
			file.write("batch %d \n" % i)

		for test in tests:
			takers = get_test_takers(i, test)
			for taker in takers:
				taker.initialize_input(test, sample_size=10**4, seed=idx, silent=True)
				t_0 = time.process_time()
				result = taker.kriging_estimate(do_mcs=False)
				t_elapsed = time.process_time()-t_0
				line = "%s,%s,%d,%f,%f,%f,%f,%d,%f\n" % (result["system"], taker.name, result["iter"], result["Pf"], result["Pfe"], result["COV"], result["re"], taker.sample_size, t_elapsed)
				print(line)
				with open("experiement%d.txt" % idx, "a") as file:
					file.write(line)
			print(datetime.datetime.now())

if __name__ == "__main__":
	run_suite = False
	if not run_suite:
		run_test_single()
	else:
		with keep.running() as k:
			for i in range(10,20):
				run_test_suite(i)

