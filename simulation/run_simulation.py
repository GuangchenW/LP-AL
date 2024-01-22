from simulation import AKMCS
from objective_functions import G_4B, G_Ras, G_oscillator, G_beam
from acquisition_functions import U, ULP, EFF, H, VAR
from evaluators import LP_Batch, NLP_Batch
from subset_samplers import U_Sampler

if __name__ == "__main__":
	run = AKMCS(acq_func=U(), batch_size=4)
	run.initialize_input(G_oscillator(), sample_size=10**5, num_init=4, random=False)
	run.kriging_estimate()
	run.tail()