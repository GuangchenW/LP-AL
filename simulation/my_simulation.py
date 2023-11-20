import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

# Convenient 2D Kriging
from pykrige.ok import OrdinaryKriging

# Density of grid for visualization
N_GRID = 400

# Monte-Carlo samples, 
N_MC=10000

with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
# for item in loaded_data:
#    print(item[0], item[1])
point_x1 = [item[0] for item in loaded_data]
point_x2 = [item[1] for item in loaded_data]

# Seems to be MC smaples alright
#plt.plot(point_x1, point_x2, 'bo')
#plt.show()

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

G = G_Ras

# This is STEP2 "...a dozen points are enough"
# Using points from MC samples instead
N_INIT = 50 # Number of bootstrap points
x1_val = point_x1[:N_INIT]
x2_val = point_x2[:N_INIT]

# Query performance function and repack data
DOE = np.zeros((N_INIT, 3))
for i in range(N_INIT):
    x1 = x1_val[i]
    x2 = x2_val[i]
    DOE[i, :] = np.array([x1, x2, G(x1, x2)])

# Learning method U
def U(mean, variance):
    var = np.maximum(variance, 0.00001)
    return abs(mean)/np.sqrt(var)

# Kriging visualization
def visualize(kriging_model, data):
    grid_x = np.linspace(-5, 5, N_GRID)
    grid_y = np.linspace(-5, 5, N_GRID)
    z, ss = kriging_model.execute('grid', grid_x, grid_y)

    plt.figure()
    contours = plt.contourf(grid_x, grid_y, z, levels=100, cmap='jet')

    plt.colorbar(label='Value')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Kriging Interpolation')
    # 绘制函数值为零的等值线
    contours = plt.contour(grid_x, grid_y, z, levels=[0], colors='r', linewidths=2)
    # 在数据点旁边标注坐标
    for x1, x2 in data[:,:2]:
        plt.text(x1, x2, f'({x1:.2f}, {x2:.2f})', fontsize=8, color='black', ha='center', va='center')
    # 添加数据点
    plt.scatter(data[:, 0], data[:, 1], c='black', label='Data')
    # 添加颜色条
    plt.colorbar(label='Value')
    # 显示图例
    plt.legend()
    plt.show()

max_iter = 100
final_kriging_model = None
for i in range(max_iter):
    # STEP3 Compute Kriging model
    kriging_model = OrdinaryKriging(
        DOE[:,0], 
        DOE[:,1], 
        DOE[:,2], 
        variogram_model="gaussian",
        variogram_parameters={'sill':5 ,'range':1.6, 'nugget':0})
    # STEP4 Estimate the probabilty of failure based on estimation of all points
    # in MC sample. P_f is calculated by P_f=N_{G<=0}/N_MC
    performance, variance = kriging_model.execute("points", point_x1, point_x2)
    P_f = np.sum(performance <= 0)/N_MC

    # STEP5 Compute learning function on the population and identify best point
    # If using U(x), G(x)-U(x)sigma(x)=0, and we want to find argmin x
    scores = U(performance, variance)
    min_id = np.argmin(scores)
    next_candidate = {
        'x1': point_x1[min_id], 
        'x2': point_x2[min_id], 
        'p': performance[min_id],
        'var': variance[min_id],
        's': scores[min_id]
    }
    next_x1 = point_x1[min_id]
    next_x2 = point_x2[min_id]
    
    print('iter ', i)
    
    # STEP6 Evaluate stopping condition
    # If min U is greater than 2, probability of making mistake on sign is 0.023 (P.6)
    final_kriging_model = kriging_model
    if next_candidate['s'] >= 2:
        print("break at ", next_candidate['s'])
        break

    # STEP7 Update DOE model
    DOE = np.append(DOE, [[next_x1, next_x2, G(next_x1, next_x2)]], axis=0)
# TODO: STEP8,9,10 for when one MC population is not enough

# STEP8: Compute coefficient of variation of the probability of failure
prediction = final_kriging_model.execute('points', point_x1, point_x2)
num_negative_predictions = np.sum(prediction[0] < 0)
P_f = np.maximum(num_negative_predictions / N_MC, 0.0001)
cov_fail = np.sqrt((1-P_f)/(P_f*N_MC))

print(f"Estimated probability of failure: {P_f:.3g}")
print(f"COV of probability of failure: {cov_fail:.3g}")


############################################################
# Visualization
grid_x = np.linspace(-5, 5, 400)
grid_y = np.linspace(-5, 5, 400)
z, ss = final_kriging_model.execute('grid', grid_x, grid_y)

plt.figure()
contours = plt.contourf(grid_x, grid_y, z, levels=100, cmap='jet')

plt.colorbar(label='Value')
plt.xlabel('X1-coordinate')
plt.ylabel('X2-coordinate')
plt.title('Kriging Interpolation')

# Level 0, the estimate of the limit state by the kriging model
contours = plt.contour(grid_x, grid_y, z, levels=[0], colors='r', linewidths=2)

# Plot the points queried
plt.scatter(DOE[:, 0], DOE[:, 1], c='black', label='Data')
# Label the points queried with their actual value
for x1, x2, h in DOE:
    plt.text(x1, x2, f'{h:.2f}', fontsize=8, color='white', ha='center', va='center')

# Kriging model contour
plt.contour(grid_x, grid_y, z, colors='white', linewidths=1, linestyles='dashed', alpha=0.5)

# Color bar and legends
plt.colorbar(label='Value')
plt.legend()

# Mesh
x1_grid, x2_grid = np.meshgrid(grid_x, grid_y)

# Query G on the grid
G_values = np.zeros((400,400))
for i in range(len(grid_x)):
    for j in range(len(grid_y)):
        G_values[i,j] = G(grid_x[i], grid_y[j])

# Actual limit state i.e. G(x1, x2)=0
contours = plt.contour(x1_grid, x2_grid, G_values, levels=[0], colors='b', linestyles='dashed')

plt.show()