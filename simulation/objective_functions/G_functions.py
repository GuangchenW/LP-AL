import numpy as np

# Example 1: 4-branch series system
def G_4B(x1, x2, k=7):
    b1 = 3 + 0.1*(x1-x2)**2 - (x1+x2)/np.sqrt(2)
    b2 = 3 + 0.1*(x1-x2)**2 + (x1+x2)/np.sqrt(2)
    b3 = (x1-x2) + k/np.sqrt(2)
    b4 = (x2-x1) + k/np.sqrt(2)
    return np.min([b1, b2, b3, b4])

# Example 2: Modified Rastrigin function
def G_Ras(x1, x2, d=10):
    def calc_term(x_i):
        return x_i**2 - 5*np.cos(2*np.pi*x_i)
    term_sum = calc_term(x1) + calc_term(x2)
    result = d - term_sum
    return result