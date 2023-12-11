import numpy as np

# Example 1: 4-branch series system
def G_4B(x, k=7):
    x1=x[0]
    x2=x[1]
    b1 = 3 + 0.1*(x1-x2)**2 - (x1+x2)/np.sqrt(2)
    b2 = 3 + 0.1*(x1-x2)**2 + (x1+x2)/np.sqrt(2)
    b3 = (x1-x2) + k/np.sqrt(2)
    b4 = (x2-x1) + k/np.sqrt(2)
    return np.min([b1, b2, b3, b4])

# Example 2: Modified Rastrigin function
def G_Ras(x, d=5):
    x1=x[0]
    x2=x[1]
    def calc_term(x_i):
        return x_i**2 - 5*np.cos(2*np.pi*x_i)
    term_sum = calc_term(x1) + calc_term(x2)
    result = d - term_sum
    return result

def G_hat(x):
    x1=x[0]
    x2=x[1]
    return 20-(x1-x2)**2-8*(x1+x2-2)**3

# 2-branch series system
def G_2B(x):
    x1=x[0]
    x2=x[1]
    b1 = 3 + 0.1*(x1-x2)**2 - (x1+x2)/np.sqrt(2)
    b2 = 3 + 0.1*(x1-x2)**2 + (x1+x2)/np.sqrt(2)
    return np.min([b1, b2])

def G_beam(x):
    w=x[0]
    b=x[1]
    L=x[2]
    E=26
    I=b**4/12
    return L/325-w*b*L**4/(8*E*I)