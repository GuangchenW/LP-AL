from simulation import AKMCS
from objective_functions import G_4B, G_Ras, G_oscillator
from acquisition_functions import ULP, NEFF, NH, VAR
from evaluators import LP_Batch, NLP_Batch
from subset_samplers import U_Sampler

if __name__ == "__main__":
	run = AKMCS(acq_func=NH(), batch_size=4)
	run.initialize_input(G_Ras(), sample_size=10**5, num_init=12, random=False)
	run.kriging_estimate()
	run.tail()