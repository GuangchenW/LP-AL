import numpy as np

from objective_functions import G_Stiffener2

if __name__ == "__main__":
    func = G_Stiffener2()
    func.estimate_failure_probability(n=3*10**5)
    
    """
    print(func._evaluate([100, 5, 5000, 5000, 37000, 23758, 5949, 16245, 10140, 19185]))
    print(func._evaluate([100, 5, 5000, 5000, 35239, 23758, 5949, 16245, 10140, 17000]))
    print(func._evaluate([110, 4.5, 5500, 5700, 35239, 23758, 5949, 16245, 10140, 15185]))
    print(func._evaluate([100, 5, 5000, 5000, 37000, 23758, 5949, 16245, 10140, 19185]))
    print(func._evaluate([120, 7, 5500, 5000, 35239, 23758, 5949, 16245, 10140, 19185]))
    print(func._evaluate([90, 4.5, 5500, 5500, 37000, 25758, 5949, 18245, 11140, 21185]))
    """