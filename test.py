import numpy as np
import matplotlib.pyplot as plt
from pykrige import OrdinaryKriging

def G(a, b):
	return a**2+b**2-4

grid_x = np.linspace(-5, 5, 400)
grid_y = np.linspace(-5, 5, 400)

x1_grid, x2_grid = np.meshgrid(grid_x, grid_y)
G_values = np.zeros((400,400))
for i in range(len(grid_x)):
    for j in range(len(grid_y)):
        G_values[i,j] = G(grid_x[i], grid_y[j])

test = np.zeros((5,3))
test[2,1]=1
for x,y in test[:,:2]:
    print(x, y)

test_data = np.array(
[[-0.27642541, -1.17201747, 10.07662985],
 [-0.70789316, -0.35510972, 4.99807863],
 [-0.28713815, 1.08399512, 11.90603058],
 [ 0.95554595, 1.7985871, 12.1610512 ],
 [ 1.87698528, 0.33048435, 7.52497452],
 [-0.47765491, 0.11012647, 8.65894864],
 [ 0.12262974, 0.41593377, 9.08118602],
 [-0.93907606, 0.36117142, 10.41042094],
 [ 1.22578394, 0.30554544, 7.45213158],
 [-0.04069968, -0.49681615, 9.58992024],
 [-0.86725536, 0.55563092, 7.60086368],
 [ 0.5831432, 0.15713839, 8.05674306],
 [-2.82325873, -1.01591449, 8.19322825]]
)

# Kriging visualization
def visualize(kriging_model, data):
    grid_x = np.linspace(-5, 5, 400)
    grid_y = np.linspace(-5, 5, 400)
    z, ss = kriging_model.execute('grid', grid_x, grid_y)
    print(np.shape(z))

    plt.figure()
    contours = plt.contourf(grid_x, grid_y, z, levels=100, cmap='jet')

    plt.colorbar(label='Value')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Kriging Interpolation')
    # 绘制函数值为零的等值线
    contours = plt.contour(grid_x, grid_y, z, levels=[0], colors='r', linewidths=2)
    # 在数据点旁边标注坐标
    for x1, x2, z in data:
        plt.text(x1, x2, f'{z:.2f}', fontsize=8, color='red', ha='center', va='center')
    # 添加数据点
    plt.scatter(data[:, 0], data[:, 1], c='black', label='Data')
    # 添加颜色条
    plt.colorbar(label='Value')
    # 显示图例
    plt.legend()
    plt.show()

kriging_model = OrdinaryKriging(
    test_data[:,0], 
    test_data[:,1], 
    test_data[:,2], 
    variogram_model="gaussian",
    variogram_parameters={'sill':5 ,'range':3, 'nugget':0})

visualize(kriging_model, test_data)
