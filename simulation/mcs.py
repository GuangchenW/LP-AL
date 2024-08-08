import numpy as np

from objective_functions import G_Stiffener

if __name__ == "__main__":
    func = G_Stiffener()
    func.estimate_failure_probability(n=10**5)