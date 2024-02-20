import torch

from simulation import AKMCS
from objective_functions import G_Simple, G_4B, G_Ras, G_Oscillator, G_Beam, G_Roof, G_Axle, G_Tube, G_High_Dim, G_FEM
from acquisition_functions import U, ULP, EFF, ERF, H, VAR
from evaluators import LP_Batch, NLP_Batch
from subset_samplers import U_Sampler

def get_test_takers(batch_size=1):
	takers = []
	takers.append(AKMCS(acq_func=U(), batch_size=batch_size))
	#takers.append(AKMCS(acq_func=ULP(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=EFF(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=ERF(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=H(), batch_size=batch_size))
	return takers

def get_test_suite():
	#return [G_4B(), G_Ras(), G_Beam(), G_Axle(), G_Oscillator(), G_Tube(), G_High_Dim()]
	return [G_4B()]

def run_test_single():
	taker = AKMCS(acq_func=EFF(), batch_size=8)
	test = G_4B()
	taker.initialize_input(test, sample_size=10**5, num_init=nearest_init_num(test.dim), silent=False)
	result = taker.kriging_estimate(do_mcs=True)
	print(result)
	taker.visualize()

def nearest_init_num(n_dim):
	num = 12
	while num < n_dim:
		num *= 2
	return num

if __name__ == "__main__":
	run_suite = True
	if not run_suite:
		run_test_single()
		exit(0)
	tests = get_test_suite()
	file = open("experiement.txt", "w")
	for i in [1,4,8]:
		file.write("batch %d \n" % i)
		takers = get_test_takers(batch_size=i)
		for taker in takers:
			for test in tests:
				taker.initialize_input(test, sample_size=10**5, num_init=nearest_init_num(test.dim))
				result = taker.kriging_estimate(do_mcs=True)
				line = "%s,%f,%s,%d,%f,%f,%f\n" % (result["system"], result["Pf"], result["name"], result["iter"], result["Pfe"], result["COV"], result["re"])
				print(line)
				file.write(line)
	file.close()

