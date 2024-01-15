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

# A new learning function for Kriging... 3 call 15+10
def G_beam(x):
    w=x[0]
    L=x[1]
    b=x[2]
    E=26
    I=(b**4)/12
    return L/325-w*b*L**4/(8*E*I)

# ESC: an efficient error-based stopping criterion 3 
def G_osc(x):
    w_0 = np.sqrt((x[0]+x[1])/x[2])
    val = 2*x[5]*np.sin(w_0*x[4]*0.5)/(x[2]*w_0**2)
    return 3*x[3]-abs(val)

# A new parallel adaptive... 3
def G_tube(x):
    t,d,L1,L2,F1,F2,P,T,S_y = x
    theta1 = 0.08726646259971647 # 5 degrees
    theta2 = 0.17453292519943295 # 10 degrees

    M = F1*L1*np.cos(theta1)+F2*L2*np.cos(theta2)
    A = np.pi/4*(d**2-(d-2*t)**2)
    c = 0.5*d
    I = np.pi/64*(d**4-(d-2*t)**4)
    J = 2*I
    tau = (T*d)/(2*J)
    sigma_x = (P+F1*np.sin(theta1)+F2*np.sin(theta2))/A + M*c/I
    sigma_max = np.sqrt(sigma_x**2+3*tau**2)

    return S_y - sigma_max
