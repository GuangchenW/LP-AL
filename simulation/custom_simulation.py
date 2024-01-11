import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import math

from ordinary_kriging import OrdinaryKriging

from objective_functions import G_4B, G_2B, G_Ras, G_hat, G_beam
from acquisition_functions import ULP, NEFF, NH
from evaluators import LP_Batch
from subset_samplers import U_Sampler

# Monte-Carlo samples
with open('../data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

#with open('./data.npy', 'rb') as file:
#    points = np.load(file)

points = np.array(loaded_data)

N_MC = np.shape(points)[0]
print(N_MC)

# Seems to be MC smaples alright
#plt.plot(point_x1, point_x2, 'bo')
#plt.show()

# Objective function
G = G_Ras

# This is STEP2 "...a dozen points are enough"
# Using points from MC samples instead
N_INIT = 12 # Number of bootstrap points

# Query performance function and repack data
DOE_input = points[:N_INIT]
DOE_output = np.zeros(N_INIT)
for i in range(N_INIT):
    DOE_output[i] = G(DOE_input[i])

max_iter = 100
kriging_model = OrdinaryKriging()
#U = Batch_Acquisition(kriging_model, utility_func="NEFF")
acq_func = ULP()
evaluator = LP_Batch(acq_func=acq_func)
sampler = U_Sampler(threshold=4)
subset_samples = []
for i in range(max_iter):

    # STEP3 Compute Kriging model
    kriging_model.train(DOE_input, DOE_output)
    # STEP4 Estimate the probabilty of failure based on estimation of all points
    # in MC sample. P_f is calculated by P_f=N_{G<=0}/N_MC


    # acquire all estimations
    mean, variance = kriging_model.execute(points)

    # sample critical region
    subset_pop, subset_mean, subset_var = sampler.sample(points, DOE_input, DOE_output, mean, variance)
    
    N_f = np.sum(mean < 0) # number of failures by Kriging model
    S_f = np.sum(subset_mean < 0) # number of likely false negatives 
    S_s = np.sum(subset_mean > 0) # number of likely false positives
    epsilon_max = max(abs(N_f/(N_f-S_f)-1), abs(N_f/(N_f+S_s)-1))
    epsilon_thr = 0.05
    print("epsilon_max : ", epsilon_max)

    # STEP6 Evaluate stopping condition
    if (epsilon_max < epsilon_thr):
        break

    # log critical region for visualization
    if len(subset_pop)>0:
        subset_samples.append(subset_pop)
    else:
        # subset empty, no more candidates left
        break

    # STEP5 Compute learning function on the population and identify best point
    # If using U(x), G(x)-U(x)sigma(x)=0, and we want to find argmin x
    #candidates = U.acquire(subset_pop, DOE_input, DOE_output, subset_mean, subset_var, 4)
    candidates = evaluator.obtain_batch(subset_pop, subset_mean, subset_var, DOE_input, DOE_output, 4)
    print("iter (%i), batch size %i" % (i, len(candidates)))

    for candidate in candidates:
        print("Selected (%.5f, %.5f) | Score : %.3f | Mean : %.3g | Var : %.3g" % (
            candidate["next"][0],
            candidate["next"][1],
            candidate["utility"],
            candidate["mean"],
            candidate["variance"]))
        # STEP7 Update DOE model
        DOE_input = np.append(DOE_input, [candidate["next"]], axis=0)
        DOE_output = np.append(DOE_output, G(candidate["next"]))
    print("--"*25)

# TODO: 9,10 for when one MC population is not enough

# STEP8: Compute coefficient of variation of the probability of failure
final_kriging_model = kriging_model
z, ss = final_kriging_model.execute(points)
num_negative_predictions = np.sum(z < 0)
P_f = np.maximum(num_negative_predictions / N_MC, 0.0001)
cov_fail = np.sqrt((1-P_f)/(P_f*N_MC))

N_true_f = 0
for i in range(N_MC):
    if G(points[i]) < 0:
        N_true_f += 1

print(f"True probability of failure: {N_true_f/N_MC:.3g}")
print(f"Estimated probability of failure: {P_f:.3g}")
print(f"COV of probability of failure: {cov_fail:.3g}")

if not np.shape(DOE_input)[1] == 2:
    exit()
############################################################
# subset sample evolution
if len(subset_samples) > 0:
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
plt.scatter(DOE_input[:, 0], DOE_input[:, 1], s=2, c='black', label='Data')
# Label the points queried with their actual value
#for x1, x2, h in DOE:
#    plt.text(x1, x2, f'{h:.2f}', fontsize=8, color='white', ha='center', va='center')

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
        G_values[i,j] = G([grid_x[i], grid_y[j]])

# Actual limit state i.e. G(x1, x2)=0
contours = plt.contour(x1_grid, x2_grid, G_values, levels=[0], colors='b', linestyles='dashed')

plt.show()

ani.save("G_Ras.gif")