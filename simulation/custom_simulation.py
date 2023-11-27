import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

from ordinary_kriging import OrdinaryKriging

from objective_functions import G_4B, G_Ras
from acquisition_functions import U_Basic

# Monte-Carlo samples
N_MC=10000

with open('../data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

point_x1 = np.array([item[0] for item in loaded_data])
point_x2 = np.array([item[1] for item in loaded_data])
points = np.dstack((point_x1,point_x2))[0]

# Seems to be MC smaples alright
#plt.plot(point_x1, point_x2, 'bo')
#plt.show()

# Objective function
G = G_Ras

# This is STEP2 "...a dozen points are enough"
# Using points from MC samples instead
N_INIT = 10 # Number of bootstrap points
x1_val = point_x1[:N_INIT]
x2_val = point_x2[:N_INIT]

# Query performance function and repack data
DOE = np.zeros((N_INIT, 3))
for i in range(N_INIT):
    x1 = x1_val[i]
    x2 = x2_val[i]
    DOE[i, :] = np.array([x1, x2, G(x1, x2)])


def U_mod(candidates, mean, variance):
    return np.array(list(map(
        lambda pair: LF2(pair[0], pair[1], pair[2]), zip(candidates, mean, variance)
        )))

def LF2(candidate, mean, variance):
    _U = U_orig_helper(mean, variance)
    min_d = np.inf
    for observation in DOE:
        dist = np.linalg.norm(candidate - observation[:2])
        min_d = dist if dist < min_d else min_d
    return _U+2/min_d

def LF(candidate, mean, variance):
    if variance < 0.001:
        return 0

    max_U = 0
    target_U = 0
    max_d = 0
    min_d = np.inf
    for observation in DOE:
        dist = np.linalg.norm(candidate - observation[:2])
        _U = U_mod_helper(candidate, observation[2], mean, variance)
        max_U = _U if _U > max_U else max_U
        max_d = dist if dist > max_d else max_d
        if dist < min_d:
            min_d = dist
            target_U = _U
    return (target_U/max_U)/(min_d/max_d)

def U_mod_helper(candidate, perform_near, mean, variance):
    denominator = np.sqrt((mean-perform_near)**2+variance)
    return abs(mean)/denominator

#U = U_mod
U = U_Basic()

max_iter = 50
final_kriging_model = None
for i in range(max_iter):
    # STEP3 Compute Kriging model
    kriging_model = OrdinaryKriging()
    kriging_model.train(DOE[:,:2], DOE[:,2])
    # STEP4 Estimate the probabilty of failure based on estimation of all points
    # in MC sample. P_f is calculated by P_f=N_{G<=0}/N_MC

    #P_f = np.sum(performance <= 0)/N_MC

    # STEP5 Compute learning function on the population and identify best point
    # If using U(x), G(x)-U(x)sigma(x)=0, and we want to find argmin x
    candidate = U.acquire(kriging_model, points, DOE[:,:2], DOE[:,2])

    next_x1 = candidate["next"][0]
    next_x2 = candidate["next"][1]
    print('iter ', i)
    print("Selected (%.10f, %.10f) | Score : %.3f | Mean : %.3f | Var : %.3f" % (
        candidate["next"][0],
        candidate["next"][1],
        candidate["utility"],
        candidate["mean"],
        candidate["variance"]))
    # STEP6 Evaluate stopping condition
    # If min U is greater than 2, probability of making mistake on sign is 0.023 (P.6)
    final_kriging_model = kriging_model
    if candidate['utility'] >= 8:
        print("break at ", candidate['utility'])
        break

    # STEP7 Update DOE model
    DOE = np.append(DOE, [[next_x1, next_x2, G(next_x1, next_x2)]], axis=0)
# TODO: STEP8,9,10 for when one MC population is not enough

# STEP8: Compute coefficient of variation of the probability of failure
z, ss = final_kriging_model.execute(points)
num_negative_predictions = np.sum(z < 0)
P_f = np.maximum(num_negative_predictions / N_MC, 0.0001)
cov_fail = np.sqrt((1-P_f)/(P_f*N_MC))

print(f"Estimated probability of failure: {P_f:.3g}")
print(f"COV of probability of failure: {cov_fail:.3g}")


############################################################
# Visualization
# Density of grid for visualization
N_GRID = 400
grid_x = np.linspace(-5, 5, N_GRID)
grid_y = np.linspace(-5, 5, N_GRID)
xpts, ypts = np.meshgrid(grid_x, grid_y)
pts = np.dstack((xpts.ravel(), ypts.ravel()))
z, ss = final_kriging_model.execute(pts)
z = z.reshape((N_GRID,N_GRID))

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
G_values = np.zeros((N_GRID, N_GRID))
for i in range(len(grid_x)):
    for j in range(len(grid_y)):
        G_values[i,j] = G(grid_x[i], grid_y[j])

# Actual limit state i.e. G(x1, x2)=0
contours = plt.contour(x1_grid, x2_grid, G_values, levels=[0], colors='b', linestyles='dashed')

plt.show()