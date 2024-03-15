from objective_functions import G_Oscillator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm


def visualize(x_mean, x_std, y_mean, y_std):
	# Visualization
	# Density of grid for visualization
	N_GRID = 400
	grid_x = np.linspace(x_mean-4*x_std, x_mean+4*x_std, N_GRID)
	grid_y = np.linspace(y_mean-4*y_std, y_mean+4*y_std, N_GRID)
	xpts, ypts = np.meshgrid(grid_x, grid_y)
	pts = np.dstack((xpts.ravel(), ypts.ravel()))

	# Mesh
	x1_grid, x2_grid = np.meshgrid(grid_x, grid_y)

	# Other constants
	c1 = 1
	c2 = 0.1
	m = 1
	r = 0.5
	t = 1
	F = 1

	# [m, r, t, F]
	variables = np.array([0.95, 0.45, 0.8, 0.8])
	step = np.array([0.05, 0.05, 0.2, 0.2]) * 0.05
	path = [variables]
	for i in range(40):
		path.append(path[-1]+step)

	frames = []
	for i in range(len(path)):
		frames.append(get_g_grid(N_GRID, grid_x, grid_y, path[i]))
	frames = np.array(frames)
	max_g = np.max(frames)
	min_g = np.min(frames)
	cmap = cm.get_cmap("jet")
	norm = plt.Normalize(min_g, max_g)
	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])

	fig, ax = plt.subplots()
	def animatef(i):
		ax.clear()
		ax.contourf(grid_x, grid_y, frames[i], levels=20, cmap=cmap, norm=norm)
		ax.contour(grid_x, grid_y, frames[i], levels=[0], colors='black', linewidths=2)
		ax.set_title("m=%.3g, r=%.3g, t=%.3g, F=%.3g"%tuple(path[i]))

	ani = animation.FuncAnimation(fig, animatef, 40, interval=200, blit=False)
	fig.supxlabel("c1")
	fig.supylabel("c2")
	fig.colorbar(sm)
	plt.show()
	ani.save("grand_tour_osc.gif")

	#animate(grid_x, grid_y, frames)

def get_g_grid(size, grid_x, grid_y, path_pt):
	g_grid = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			c1 = grid_x[i]
			c2 = grid_y[j]

			m, r, t, F = path_pt
			w_0 = np.sqrt((c1+c2)/m)
			val = 2*F*np.sin(w_0*t*0.5)/(m*w_0**2)
			g_grid[i,j] = 3*r-abs(val)
	return g_grid

def animate(grid_x, grid_y, frames):
	fig, ax = plt.subplots()
	artists = []
	for i in range(len(frames)):
		cmap = ax.contourf(grid_x, grid_y, frames[i], levels=100, cmap='jet')
		#ls = ax.contour(grid_x, grid_y, frames[i], levels=[0], colors='black', linewidths=2)
		artists.append(cmap)
	ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=200)


if __name__ == "__main__":
	visualize(1, 0.1, 0.1, 0.01)
