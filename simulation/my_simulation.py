import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from objective_functions import G_4B

matplotlib.rcParams["mathtext.fontset"]="cm"

func = G_4B()
N_GRID = 400
grid_x = np.linspace(-5, 5, N_GRID)
grid_y = np.linspace(-5, 5, N_GRID)
xpts, ypts = np.meshgrid(grid_x, grid_y)


fig, ax = plt.subplots()
# Mesh
x1_grid, x2_grid = np.meshgrid(grid_x, grid_y)

# Query G on the grid
G_values = np.zeros((N_GRID, N_GRID))
for i in range(len(grid_x)):
    for j in range(len(grid_y)):
        G_values[i,j] = func.evaluate([grid_x[i], grid_y[j]])

# Actual limit state i.e. G(x1, x2)=0
contours = ax.contourf(x1_grid, x2_grid, G_values, levels=100, cmap='YlGn_r')
ax.contour(x1_grid, x2_grid, G_values, levels=[0], colors='b')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(contours, label="value", cax=cax)
ax.set_aspect(0.75)
cax.set_aspect(15)
ax.set_xlabel(r"$x_1$", fontsize=12)
ax.set_ylabel(r"$x_2$", fontsize=12)

handles, _ = ax.get_legend_handles_labels()
actual_limit_state = Line2D([0], [0], label=r"$f(\mathbf{x})=0$", color="blue")
handles.extend([actual_limit_state])
ax.legend(handles=handles, loc=1, prop={"size":10})

plt.tight_layout()
plt.savefig("test.png", dpi=300)