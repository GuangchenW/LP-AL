from simulation.utility_functions import U

class UPE(AcquisitionFunction):
    def __init__(self, device="cpu"):
        self.name = "UPE"
        super().__init__(name=self.name, device=device)

    def acquire(
        self,
        input_population,
        doe_input,
        doe_response,
        mean,
        variance,
        n_points=1
    ):