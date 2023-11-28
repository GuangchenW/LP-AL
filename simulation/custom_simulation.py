import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import math

from ordinary_kriging import OrdinaryKriging

from objective_functions import G_4B, G_Ras, G_hat
from acquisition_functions import Single_Acquisition, Batch_Acquisition
from subset_samplers import U_Sampler

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
G = G_hat

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

U = Batch_Acquisition(utility_func="ULP")
sampler = U_Sampler(threshold=6)

max_iter = 100
final_kriging_model = None
kriging_model = OrdinaryKriging()
subset_samples = []
p_failures = []
for i in range(max_iter):

    # STEP3 Compute Kriging model
    kriging_model.train(DOE[:,:2], DOE[:,2])
    final_kriging_model = kriging_model
    # STEP4 Estimate the probabilty of failure based on estimation of all points
    # in MC sample. P_f is calculated by P_f=N_{G<=0}/N_MC

    # STEP6 Evaluate stopping condition
    # If min U is greater than 2, probability of making mistake on sign is 0.023 (P.6)
    if (len(p_failures) > 2 and 
        abs(p_failures[-1]-p_failures[-2])/p_failures[-1] < 0.001 and
        abs(p_failures[-2]-p_failures[-3])/p_failures[-2] < 0.001 
        ):
        print(p_failures[-3:])
        print("break at ", candidate['utility'])
        break

    mean, variance = kriging_model.execute(points)
    P_f = np.sum(mean <= 0)/N_MC
    p_failures.append(P_f)

    # sample critical region
    subset_pop, subset_mean, subset_var = sampler.sample(points, DOE[:,:2], DOE[:,2], mean, variance)
    
    # log critical region for visualization
    if len(subset_pop)>0:
        subset_samples.append(subset_pop)
    else:
        # subset empty, no more candidates left
        break

    # STEP5 Compute learning function on the population and identify best point
    # If using U(x), G(x)-U(x)sigma(x)=0, and we want to find argmin x
    candidates = U.acquire(subset_pop, DOE[:,:2], DOE[:,2], subset_mean, subset_var, 4)
    print("iter (%i), batch size %i" % (i, len(candidates)))

    for candidate in candidates:
        next_x1 = candidate["next"][0]
        next_x2 = candidate["next"][1]
        print("Selected (%.5f, %.5f) | Score : %.3f | Mean : %.3f | Var : %.3f" % (
            candidate["next"][0],
            candidate["next"][1],
            candidate["utility"],
            candidate["mean"],
            candidate["variance"]))
        # STEP7 Update DOE model
        DOE = np.append(DOE, [[next_x1, next_x2, G(next_x1, next_x2)]], axis=0)
    print("--"*25)

# TODO: 9,10 for when one MC population is not enough

# STEP8: Compute coefficient of variation of the probability of failure
z, ss = final_kriging_model.execute(points)
num_negative_predictions = np.sum(z < 0)
P_f = np.maximum(num_negative_predictions / N_MC, 0.0001)
cov_fail = np.sqrt((1-P_f)/(P_f*N_MC))

print(f"Estimated probability of failure: {P_f:.3g}")
print(f"COV of probability of failure: {cov_fail:.3g}")

############################################################
# subset sample evolution
fig, ax = plt.subplots()
artists = []
for i in range(len(subset_samples)):
    samples = np.array(subset_samples[i]).T
    container = ax.scatter(samples[0],samples[1], c="b")
    txt = ax.text(0.05,0.05, str(i), ha="right", va="bottom", transform=fig.transFigure)
    artists.append([container, txt])
ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=8000/len(subset_samples))
plt.show()


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

ani.save("G_Ras.gif")
