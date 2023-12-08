import numpy as np

from .acquisition_functions import AcquisitionFunction

class Batch_Acquisition(AcquisitionFunction):
    def __init__(self, model, utility_func="ULP", device="cpu"):
        #if not utility_func in ["ULP"]:
        #    raise Exception("utility_func {utility_func} not supported for batch acquisition")
        self.model = model
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
            k=i+1
            mean, variance = self.model.fantasize(doe_input[-k:], doe_response[-k:], input_population)

            #m, v = self.model.fantasize([[1,1],[2,2]], [10,5], [[1,1],[2,2]])
            #print(m,v)

        return np.array(out)
