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


plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 20
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
#report
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.rcParams["font.size"] = 15
# plt.rc('xtick', labelsize=12)
# plt.rc('ytick', labelsize=12)

random_points = []
X = []
for dim in np.arange(2):
    # random_data_point_each_dim = np.random.uniform(0, 1, 3).reshape(1, 3)
    random_data_point_each_dim = np.linspace(0, np.pi, 100).reshape(1, 100)
    random_points.append(random_data_point_each_dim)
# Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
random_points = np.vstack(random_points)

# Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
for sample_num in np.arange(100):
    array = []
    for dim_count in np.arange(2):
        array.append(random_points[dim_count, sample_num])
    X.append(array)


X = np.vstack(X)

print(X)
x1 = X[:, 0]
x2 = X[:, 1]
print(x1,"\n\n", x2)
X1, X2 = np.meshgrid(x1, x2)
y = np.sin(X1) * ((np.sin(((X1**2)/np.pi))) ** 20) + np.sin(X2) * ((np.sin((2*(X2**2)/np.pi))) ** 20)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, y, rstride=1, cstride=1,
                       cmap='viridis',linewidth=1, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
fig.colorbar(surf, shrink=0.5, aspect=20, pad = 0.05)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
# plt.xticks(rotation=-45)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Output, f(x)')

plt.title("Michalewicz 2D Function")
fig.savefig("mic2d_fun.pdf", pad_inches=0, bbox_inches='tight')
fig.savefig("mic2d_fun.eps", pad_inches=0, bbox_inches='tight')
plt.autoscale(tight=True)
plt.show()
exit(0)