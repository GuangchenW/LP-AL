import numpy as np

from .acquisition_functions import AcquisitionFunction

class Batch_Acquisition(AcquisitionFunction):
    def __init__(self, utility_func="ULP", device="cpu"):
        if not utility_func in ["ULP"]:
            raise Exception("utility_func {utility_func} not supported for batch acquisition")
        super().__init__(utility_func=utility_func, device=device)

    def acquire(
        self,
        input_population,
        doe_input,
        doe_response,
        mean,
        variance,
        n_points
    ):
        out = []
        print("subset size", len(input_population))
        for i in range(min(n_points, len(input_population))):
            utilities = self.utility_func(input_population, mean, variance, doe_input, doe_response)

            min_id = np.argmin(utilities)
            out.append({
            "next": input_population[min_id],
            "mean": mean[min_id],
            "variance": variance[min_id],
            "utility": utilities[min_id]
            })

            doe_input = np.append(doe_input, [input_population[min_id]], axis=0)
            doe_response = np.append(doe_response, [mean[min_id]])

        return np.array(out)
