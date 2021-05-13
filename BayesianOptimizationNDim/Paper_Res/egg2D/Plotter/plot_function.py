import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
import scipy.optimize as opt
import scipy as sp
import matplotlib.pyplot as plt
# from numpy import *
import numpy as np
import datetime
from scipy.spatial import distance
import collections
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import plotly.plotly as py
import plotly.graph_objs as go
# from sklearn.preprocessing import MinMaxScaler
# Testing different 2D functions
number_of_dimensions = 2
bounds =[[-512,512], [-512,512]]
number_of_observed_samples = 200


# Data structure to hold the data
random_points = []
X = []

# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    # random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],number_of_observed_samples).reshape(1,number_of_observed_samples)
    random_data_point_each_dim = np.linspace(bounds[dim][0], bounds[dim][1],number_of_observed_samples).reshape(1,number_of_observed_samples)
    random_points.append(random_data_point_each_dim)

# Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
random_points = np.vstack(random_points)

# Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
for sample_num in np.arange(number_of_observed_samples):
    array = []
    for dim_count in np.arange(number_of_dimensions):
        array.append(random_points[dim_count, sample_num])
    X.append(array)
X = np.vstack(X)


x1 = X[:, 0]
x2 = X[:, 1]
X1, X2 = np.meshgrid(x1, x2)

# Eggholder function
y = -1 * (-(X2 + 47) * np.sin(np.sqrt(np.abs(X2 + (X1/2) + 47))) - X1 * np.sin(np.sqrt(np.abs(X1 - (X2+47)))))


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, y, rstride=1, cstride=1,
                       cmap='viridis',linewidth=1, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
fig.colorbar(surf, shrink=0.5, aspect=20)


# xs1 = x1
# xs2 = x2
# xss1 = []
# xss2 = []
# ys = []
# for i in xs1:
#     for j in xs2:
#         xss1.append(i)
#         xss2.append(j)
#         ys.append(z_func(i,j))
# print(xss1, "\n", xss2, "\n", ys)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.scatter(X1,X2,z, c='r', marker="o",linewidth=0.5);
# ax.scatter(xss1,xss2,ys, c=ys, cmap='jet', marker="o",linewidth=0.5);


ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Output, f(x)')
plt.show()
exit(0)
