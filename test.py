import numpy as np
import matplotlib.pyplot as plt

def test(a, b):
	return a+b

grid_x = np.linspace(-5, 5, 400)
grid_y = np.linspace(-5, 5, 400)

x1_grid, x2_grid = np.meshgrid(grid_x, grid_y)
G_values = test(grid_x, grid_y)

# 绘制等值线图
contours = plt.contour(x1_grid, x2_grid, G_values, levels=[0], colors='b', linestyles='dashed')