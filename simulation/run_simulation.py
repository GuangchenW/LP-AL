import torch

from simulation import AKMCS
from objective_functions import G_4B, G_Ras, G_oscillator, G_beam, G_Roof, G_axle, G_tube, G_High_Dim
from acquisition_functions import U, ULP, EFF, H, VAR
from evaluators import LP_Batch, NLP_Batch
from subset_samplers import U_Sampler

if __name__ == "__main__":
	run = []
	#run.append(AKMCS(acq_func=EFF(), batch_size=1))
	run.append(AKMCS(acq_func=EFF(), batch_size=4))
	#run.append(AKMCS(acq_func=H(), batch_size=1))
	#run.append(AKMCS(acq_func=H(), batch_size=4))
	#run.append(AKMCS(acq_func=H(), batch_size=8))
	#tests = [G_4B(), G_Ras(), G_beam(), G_axle(), G_tube(), G_High_Dim()]
	tests = [G_beam()]
	for test in tests:
		for r in run:
			r.initialize_input(test, sample_size=10**5, num_init=12)
			r.kriging_estimate()
			r.tail()