import torch

from simulation import AKMCS
from objective_functions import G_Simple, G_4B, G_Ras, G_Oscillator, G_Beam, G_Roof, G_Axle, G_Tube, G_High_Dim
from acquisition_functions import U, ULP, EFF, H, VAR
from evaluators import LP_Batch, NLP_Batch
from subset_samplers import U_Sampler

def get_test_takers(batch_size=1):
	takers = []
	takers.append(AKMCS(acq_func=U(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=ULP(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=EFF(), batch_size=batch_size))
	takers.append(AKMCS(acq_func=H(), batch_size=batch_size))
	return takers

def get_test_suite():
	return [G_4B(), G_Ras(), G_Beam(), G_Axle(), G_Oscillator(), G_Tube(), G_High_Dim()]

def run_test_single():
	taker = AKMCS(acq_func=EFF(), batch_size=8)
	test = G_Oscillator()
	taker.initialize_input(test, sample_size=10**5, num_init=12, silent=False)
	taker.kriging_estimate()
	taker.visualize()

if __name__ == "__main__":
	run_suite = False
	if not run_suite:
		run_test_single()
		exit(0)
	tests = get_test_suite()
	for i in [1,4,8]:
		takers = get_test_takers(batch_size=i)
		for taker in takers:
			for test in tests:
				taker.initialize_input(test, sample_size=10**5, num_init=12)
				taker.kriging_estimate()

