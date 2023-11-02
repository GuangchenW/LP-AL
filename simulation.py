import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

from pykrige import OrdinaryKriging

# BO的结果
# MC抽样得到两个随机变量的大样本数据，代码在MC-P.py
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
# for item in loaded_data:
#    print(item[0], item[1])
point_x = [item[0] for item in loaded_data]
point_y = [item[1] for item in loaded_data]


# 计算初始DoE
def G(x1, x2):
    d = 10  # 假设 d 的值为 10
    term_sum = (x1 ** 2 - 5 * np.cos(2 * np.pi * x1))+(x2 ** 2 - 5 * np.cos(2 * np.pi * x2))
    result = d - term_sum
    return result


# 输入两个变量的值
# Bootstrap how?
x1_values = [-2.13794975764486, 0.04445213253578309, 0.3631411537417851, 0.030631120880670028, 1.1839697582628255]
x2_values = [-0.2013680927242529, -1.6083640083357895, 1.665154304490278, 2.2708314089194266, -0.055223676976110435]
# 计算和输出 G(x1, x2) 的值
result_array = np.empty((5, 3))  # 创建一个5行3列的空数组
for i, (x1, x2) in enumerate(zip(x1_values, x2_values)):
    result = G(x1, x2)
    result_array[i, 0] = x1  # 第一列是 x1
    result_array[i, 1] = x2  # 第二列是 x2
    result_array[i, 2] = result  # 第三列是计算得到的 result

# 输入数据,将数据拆分为位置坐标和对应的值
data = result_array

# 外循环次数，总的取点数是外循环次数乘以内循环取点数
# ?? Where convergence condition
num_iter = 40

DOE_data = data.copy()

for iteration in range(num_iter):
    DOE_locations = DOE_data[:, :2]
    DOE_values = DOE_data[:, 2]
    # !!Need to use Gaussian model
    kriging_model = OrdinaryKriging(DOE_locations[:, 0], DOE_locations[:, 1], DOE_values)

    ''' #不需要展示过程图像时，以下代码注释
   grid_x = np.linspace(-5, 5, 400)
    grid_y = np.linspace(-5, 5, 400)
    z, ss = kriging_model.execute('grid', grid_x, grid_y)
    projection, _ = kriging_model.execute('grid', grid_x, grid_y)

    plt.figure()
    contours = plt.contourf(grid_x, grid_y, z, levels=100, cmap='jet')

    plt.colorbar(label='Value')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Kriging Interpolation')
    # 绘制函数值为零的等值线
    contours = plt.contour(grid_x, grid_y, z, levels=[0], colors='r', linewidths=2)
    # 在数据点旁边标注坐标
    for x, y in zip(DOE_data[:, 0], DOE_data[:, 1]):
        plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=8, color='black', ha='center', va='center')
    # 绘制Kriging模型的边界
    plt.contour(grid_x, grid_y, projection, colors='white', linewidths=1, linestyles='dashed', alpha=0.5)
    # 添加数据点
    plt.scatter(DOE_data[:, 0], DOE_data[:, 1], c='black', label='Data')
    plt.scatter(DOE_data[-5:, 0], DOE_data[-5:, 1], c='white', label='Data')
    # 添加颜色条
    plt.colorbar(label='Value')
    # 显示图例
    plt.legend()
    plt.show()'''

    print(f"Iteration {iteration + 1}:")
    # 内循环取点数
    num_iterations = 10
    Virtual_data = DOE_data.copy()

    for iteration in range(num_iterations):
        Virtual_locations = Virtual_data[:, :2]
        Virtual_values = Virtual_data[:, 2]
        Virtual_kriging_model = OrdinaryKriging(Virtual_locations[:, 0], Virtual_locations[:, 1], Virtual_values)
        # 获取所有点处的均值和方差
        means, variances = Virtual_kriging_model.execute('points', point_x, point_y)
        # 计算 U 函数值
        # !!! THIS IS NOT RIGHT! The u_values here are computed as lower confidence bounds,
        # which are used for optimizations. The problem here is reliability, so 
        # U is defined as mean-U*std=lcb=0, because we are interested in the limit state 0.
        # So we want to find x such that U is minimized, i.e. It takes a small deviation
        # for point x to cross the limit state. So U=G/std would be correct
        u_values = means - 2.5 * np.sqrt(variances)
        # 找到具有最小 U 函数值的点的索引
        min_u_index = np.argmin(u_values)
        # 输出最小 U 函数值对应的 x 和 y
        min_u_x = point_x[min_u_index]
        min_u_y = point_y[min_u_index]
        # print(f"Iteration {iteration + 1}:")
        # print("最小的U_function对应的坐标：")
        # print("x =", min_u_x)
        # print("y =", min_u_y)

        # 计算预测值
        prediction = kriging_model.execute('points', np.array([min_u_x]), np.array([min_u_y]))
        # 添加新的数据点
        Virtual_new_data = np.array([[min_u_x, min_u_y, prediction[0][0]]])
        # 将新的数据点合并到原有数据中
        Virtual_data = np.vstack((Virtual_data, Virtual_new_data))

    # 输出最终的新增点数据

    print(Virtual_data[-num_iterations:])

    # 更改内环取点个数需更改以下三行代码中出现的5或10
    # 添加新的数据点
    x1_new = Virtual_data[-10:, 0]
    x2_new = Virtual_data[-10:, 1]

    # 计算和输出 G(x1, x2) 的值
    new_data = np.empty((10, 3))  # 创建一空数组
    for i, (x1, x2) in enumerate(zip(x1_new, x2_new)):
        result = G(x1, x2)
        new_data[i, 0] = x1
        new_data[i, 1] = x2
        new_data[i, 2] = result

    # 将新的数据点合并到原有数据中
    DOE_data = np.vstack((DOE_data, new_data))
    # print(DOE_data)

kriging_model = OrdinaryKriging(DOE_data[:, 0], DOE_data[:, 1], DOE_data[:, 2])
grid_x = np.linspace(-5, 5, 400)
grid_y = np.linspace(-5, 5, 400)
z, ss = kriging_model.execute('grid', grid_x, grid_y)
projection, _ = kriging_model.execute('grid', grid_x, grid_y)

plt.figure()
contours = plt.contourf(grid_x, grid_y, z, levels=100, cmap='jet')

plt.colorbar(label='Value')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Kriging Interpolation')
# 绘制函数值为零的等值线
contours = plt.contour(grid_x, grid_y, z, levels=[0], colors='r', linewidths=2)
# 在数据点旁边标注坐标
for x, y in zip(DOE_data[:, 0], DOE_data[:, 1]):
    plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=8, color='black', ha='center', va='center')
# 绘制Kriging模型的边界
plt.contour(grid_x, grid_y, projection, colors='white', linewidths=1, linestyles='dashed', alpha=0.5)
# 添加数据点
plt.scatter(DOE_data[:, 0], DOE_data[:, 1], c='black', label='Data')
plt.scatter(DOE_data[-5:, 0], DOE_data[-5:, 1], c='white', label='Data')
# 添加颜色条
plt.colorbar(label='Value')
# 显示图例
plt.legend()

# 创建网格点
x1_grid, x2_grid = np.meshgrid(grid_x, grid_y)

# 计算G的值
G_values = G(x1_grid, x2_grid)

# 绘制等值线图
contours = plt.contour(x1_grid, x2_grid, G_values, levels=[0], colors='b', linestyles='dashed')

plt.show()  # 与tuxiang.py中的真实图像相对比

# 预测新的数据点的值
prediction = kriging_model.execute('points', point_x, point_y)
# 统计小于0的个数
num_negative_predictions = np.sum(prediction[0] < 0)

# 计算比例
ratio = num_negative_predictions / len(prediction[0])

print(f"小于0的预测值比例：{ratio:.2%}")