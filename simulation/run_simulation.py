import torch

from simulation import AKMCS
from objective_functions import G_4B, G_Ras, G_oscillator, G_beam, G_tube
from acquisition_functions import U, ULP, EFF, H, VAR
from evaluators import LP_Batch, NLP_Batch
from subset_samplers import U_Sampler

if __name__ == "__main__":
	run = AKMCS(acq_func=ULP(), batch_size=4)
	#run.initialize_input(G_tube(), sample_size=10**5, num_init=12, lengthscale=torch.Tensor([1,4,1,1,100,100,1000,1000,3]))
	run.initialize_input(G_Ras(), sample_size=10**5, num_init=12)
	run.kriging_estimate()
	run.tail()