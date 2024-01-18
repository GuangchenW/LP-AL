from simulation import AKMCS
from objective_functions import G_4B, G_Ras
from acquisition_functions import ULP, NEFF, NH, VAR
from evaluators import LP_Batch, NLP_Batch
from subset_samplers import U_Sampler

if __name__ == "__main__":
	run = AKMCS(acq_func=NH(), batch_size=8)
	run.initialize_input(G_Ras(), num_init=12)
	run.kriging_estimate()
	run.tail()