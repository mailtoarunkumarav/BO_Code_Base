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
#

# Scatter 3D function and contour plot for syn2d.
def z_func(x, y):
    return (np.exp(-x)*np.sin(2*np.pi*x)) * (np.exp(-y)*np.sin(2*np.pi*y))

random_points = []
X = []
for dim in np.arange(2):
    # random_data_point_each_dim = np.random.uniform(0, 1, 3).reshape(1, 3)
    random_data_point_each_dim = np.linspace(0, 4, 100).reshape(1, 100)
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
x1 = X[:, 0]
x2 = X[:, 1]
X1, X2 = np.meshgrid(x1, x2)
y = (np.exp(-X1)*np.sin(2*np.pi*X1)) * (np.exp(-X2)*np.sin(2*np.pi*X2))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, y, rstride=1, cstride=1,
                       cmap='viridis',linewidth=1, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
fig.colorbar(surf, shrink=0.5, aspect=20)


plt.show()
exit(0)


# def len_scale_func_gauss(data_point_value, len_scale):
#     bias = 0.01
#     mean = len_scale[0]
#     std_dev = len_scale[1]
#
#     exp_term = np.exp((-0.5) * (((data_point_value - mean) / std_dev) ** 2))
#     pre_term = 1 / np.sqrt(2 * np.pi * (std_dev ** 2))
#     value = pre_term * exp_term + bias
#
#     if value == 0:
#         value = 1e-6
#
#     return value
#
#
# number_of_dimensions = 2
# data_point1 = np.array([[0.94763226, 0.42830868], [0.22654742 ,0.76414069], [0.59442014, 0.00286059]])
# data_point2 = np.array([[0.94763226, 0.42830868], [0.22654742 ,0.76414069], [0.59442014, 0.00286059]])
# len_scale_params = [[0.47148474, 0.38637105],[0.41093924, 0.3367968 ]]
# len_scale_func_type =['gaussian', 'gaussian']
# signal_variance = 0.5145695683100672
# kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
#
# for i in np.arange(len(data_point1)):
#     for j in np.arange(len(data_point2)):
#         len_scale_vectors = []
#         for d in np.arange(number_of_dimensions):
#
#             if len_scale_func_type[d] == 'gaussian':
#                 len_scale_vector_datapoint1 = len_scale_func_gauss(data_point1[i][d], len_scale_params[d])
#                 len_scale_vector_datapoint2 = len_scale_func_gauss(data_point2[j][d], len_scale_params[d])
#
#
#             len_scale_vectors.append([len_scale_vector_datapoint1, len_scale_vector_datapoint2])
#
#         difference = data_point1[i, :] - data_point2[j, :]
#         total_product = 1
#         total_sum = 0
#
#         for k in np.arange(number_of_dimensions):
#             denominator = len_scale_vectors[k][0] ** 2 + len_scale_vectors[k][1] ** 2
#             total_product *= (2 * len_scale_vectors[k][0] * len_scale_vectors[k][1]) / denominator
#             total_sum += 1 / denominator
#         if (total_product < 0):
#             print("Product term of length scale is less than zero", data_point1, data_point2)
#
#         squared_diff = np.dot(difference, difference.T)
#         each_kernel_val = (signal_variance ** 2) * np.sqrt(total_product) * (np.exp((-1) * squared_diff * total_sum))
#         kernel_mat[i, j] = each_kernel_val
#
# print(kernel_mat)

# exit(0)
X1 = -442.41777858
X2 =  -366.86402399
yy = -1 * (-(X2 + 47) * np.sin(np.sqrt(np.abs(X2 + (X1 / 2) + 47))) - X1 * np.sin(np.sqrt(np.abs(X1 - (X2 + 47)))))
print("fin", yy)
exit(0)




# Testing different 2D functions
number_of_dimensions = 2
bounds =[[-10,10], [-10,10]]
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


# # # Grid Search Maxima
# maxx = -1 * float("inf")
# val=[]
# x1 = np.linspace (400,512,1000)
# x2 = np.linspace (350,450,1000)
#
# for X1 in x1:
#     for X2 in x2:
#         # print("X1",X1, "  X2",X2)
#         yy = -1*(-(X2 + 47) * np.sin(np.sqrt(np.abs(X2 + (X1 / 2) + 47))) - X1 * np.sin(np.sqrt(np.abs(X1 - (X2 + 47)))))
#         if yy > maxx:
#             maxx = yy
#             val = np.array([X1,X2,yy])
#
#
#
#
# print("final Max: ", val)
# X1 = 512
# X2 = 404.254
# yy = -1 * (-(X2 + 47) * np.sin(np.sqrt(np.abs(X2 + (X1 / 2) + 47))) - X1 * np.sin(np.sqrt(np.abs(X1 - (X2 + 47)))))
# print("fin", yy)
# exit(0)


# print (X)

# ##Normalizing code
# Xmin = np.array([1])
# Xmax = np.array([5])
# den = Xmax - Xmin
# num = (X - Xmin)
# X1 = np.divide(num,den)
# print("\n\n",X1,"\n")
# xnew = np.array(X1[8])
# xorig = np.multiply(xnew.T,(Xmax-Xmin))+Xmin
# print(X1[8], " OriginalL: ", xorig)
#
x1 = X[:, 0]
x2 = X[:, 1]
X1, X2 = np.meshgrid(x1, x2)

# Eggholder function
# y = -1 * (-(X2 + 47) * np.sin(np.sqrt(np.abs(X2 + (X1/2) + 47))) - X1 * np.sin(np.sqrt(np.abs(X1 - (X2+47)))))
# Levy function
w1= 1 + ((X1-1)/4)
w2= 1 + ((X2-1)/4)
y =  -1 *(((np.sin(np.pi*w1))**2) + ((w1-1)**2) * (1+10*((np.sin((np.pi * w1) + 1))**2))  +  ((w2-1)**2)*(1+((np.sin(2*np.pi*w2))**2)))
# y =  (np.exp(-X1) * np.sin(2 * np.pi * X1)) * (np.exp(-X2) * np.sin(2 * np.pi * X2))

# Ackley function
# a = 20
# b = 0.2
# c = 2 * np.pi
# y= -1*(-20* np.exp(-0.2*np.sqrt(0.5 * (X1 **2 + X2 **2))) - np.exp(0.5*( np.cos(2*np.pi*X1) + np.cos(2*np.pi*X2))) + 20 + np.exp(1))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, y, rstride=1, cstride=1,
                       cmap='viridis',linewidth=1, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
fig.colorbar(surf, shrink=0.5, aspect=20)


plt.show()
exit(0)


#Ackley 1D function
x = np.linspace(-40,40,1000)
y = -1 * (- 20 * np.exp(-0.2 * np.sqrt(x*x) ) - (np.exp(np.cos(2 * np.pi * x  ))) + 20 + np.exp(1))
plt.plot(x,y)
plt.show()
exit(0)



#Standardizing values
# x = np.random.uniform(0.5,2.5,1000)
# x.sort()
x = np.linspace(0.5,2.5,1000)
print("\n\n",x)
# y = np.linspace(0,3,100)
y =  -1 * ((((np.sin(10 * np.pi * x))/(2*(x)))) +(x-1) ** 4)
plt.figure("1")
plt.clf
plt.plot(x,y, "g" )
print("y_original", y)
plt.figure("2")
plt.clf
x = (x-0.5)/2
print(np.mean(y), np.std(y))
y = (y - np.mean(y))/ (np.std(y))
print(y)
plt.plot(x,y, "r" )

plt.show()

# # scaler = MinMaxScaler()
# # scaler.fit(x)
# # x = scaler.transform(x)
# x = x/10
# # y = np.exp(-x) * np.sin(x)

exit(0)

#updated 3D plot

random_points = []
X = []
for dim in np.arange(2):
    # random_data_point_each_dim = np.random.uniform(0, 1, 3).reshape(1, 3)
    random_data_point_each_dim = np.linspace(0.5, 2.5, 100).reshape(1, 100)
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
y = (np.exp(-X1) * np.sin(2 * np.pi * X1)) * (np.exp(-X2) * np.sin(2 * np.pi * X2))
# y = 100 * (X2- X1**3)**2 + (1-X1)**2
# grid of point
# print(y)
# xx = 0.233
# yy = 0.218
# ymax = (np.exp(-xx) * np.sin(2 * np.pi * xx)) * (np.exp(-yy) * np.sin(2 * np.pi * yy))
# print(ymax)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, y, rstride=1, cstride=1,
    cmap='viridis', linewidth=1, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
fig.colorbar(surf, shrink=0.5, aspect=20)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, y , rstride=1, cstride=1,
                       cmap='viridis',linewidth=1, antialiased=False)
                      # cmap=cm.RdBu,linewidth=0, antialiased=False)
                      #  cmap='viridis',linewidth=0, antialiased=False)
                      #  cmap=plt.cm.jet,linewidth=0, antialiased=False)
                      #  cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
fig.colorbar(surf, shrink=0.5, aspect=20)

plt.show()
exit(0)


# Scatter 3D function and contour plot for syn2d.
def z_func(x, y):
    return (np.exp(-x)*np.sin(2*np.pi*x)) * (np.exp(-y)*np.sin(2*np.pi*y))

random_points = []
X = []
for dim in np.arange(2):
    # random_data_point_each_dim = np.random.uniform(0, 1, 3).reshape(1, 3)
    random_data_point_each_dim = np.linspace(0, 3, 100).reshape(1, 100)
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
x1 = X[:, 0]
x2 = X[:, 1]
X1, X2 = np.meshgrid(x1, x2)
yy = z_func(X1,X2)
x = np.arange(0.5, 2.5, 0.001)
y = np.arange(0.5, 2.5, 0.001)
xx, yy = np.meshgrid(x, y, sparse=True)
z = (np.exp(-xx)*np.sin(2*np.pi*xx)) * (np.exp(-yy)*np.sin(2*np.pi*yy))
zz = (np.exp(-0.224174)*np.sin(2*np.pi*0.224174)) * (np.exp(-0.223211)*np.sin(2*np.pi*0.223211))
print(zz)
h = plt.contourf(x,y,z, cmap='jet')
# plt.show()
#
# plt.contourf(X1,X2,yy)

xs1 = x1
xs2 = x2
xss1 = []
xss2 = []
ys = []
for i in xs1:
    for j in xs2:
        xss1.append(i)
        xss2.append(j)
        ys.append(z_func(i,j))
# print(xss1, "\n", xss2, "\n", ys)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.scatter(X1,X2,z, c='r', marker="o",linewidth=0.5);
# ax.scatter(xss1,xss2,ys, c=ys, cmap='jet', marker="o",linewidth=0.5);

plt.show()
exit(0)


# Normalizing in multiple dimensions

number_of_dimensions = 1
bounds =[[1,4]]
number_of_observed_samples = 10

# Data structure to hold the data
random_points = []
X = []

# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],number_of_observed_samples).reshape(1,number_of_observed_samples)
    # random_data_point_each_dim = np.linspace(bounds[dim][0], bounds[dim][1],number_of_observed_samples).reshape(1,number_of_observed_samples)
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

print (X)



Xmin = np.array([1])
Xmax = np.array([5])
den = Xmax - Xmin
num = (X - Xmin)
X1 = np.divide(num,den)
print("\n\n",X1,"\n")
xnew = np.array(X1[8])
xorig = np.multiply(xnew.T,(Xmax-Xmin))+Xmin
print(X1[8], " OriginalL: ", xorig)







exit(0)




# Trying 3D plots
# def z_func(x, y):
#     return (np.exp(-x)*np.sin(2*np.pi*x)) * (np.exp(-y)*np.sin(2*np.pi*y))
#
#
# x = np.linspace(0,3,100)
# y = np.linspace(0,3,100)
# X, Y = np.meshgrid(x, y)  # grid of point
# Z = z_func(X, Y)  # evaluation of the function on the grid
#

### Surface
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                        cmap='viridis',linewidth=1, antialiased=False)
#                       # cmap=cm.RdBu,linewidth=0, antialiased=False)
#                       #  cmap='viridis',linewidth=0, antialiased=False)
#                       #  cmap=plt.cm.jet,linewidth=0, antialiased=False)
#                       #  cmap=plt.cm.coolwarm,linewidth=0, antialiased=False)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=20)

###Wireframe
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(X, Y, Z, color='black')
# ax.set_title('wireframe');

### contour
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');

plt.show()



exit(0)

#testing acq functions
# [1.52370996]  --  [[1.57859025]]  , mean=  [[2.19446174]] , std= [0.08309777]  ymax= 0.6058714922092715
# mean=  [[]] , std= []  ymax=
mean=  2.19446174
std_dev= 0.08309777
y_max= 0.6058714922092715
epsilon2 = 0.01
z_value = (mean - y_max - epsilon2) / std_dev
print("z", z_value)
zpdf = norm.pdf(z_value)
zcdf = norm.cdf(z_value)
print(zpdf, zcdf)
ei_acq_func = np.dot(zcdf, (mean - y_max - epsilon2)) + np.dot(std_dev, zpdf)
print(ei_acq_func)
exit(0)


#Implementing linearly varying length scale for multiple dimensions
len_scale_params = [[1,2],[1,2,3]]
len_scale_func_type = ['linear', 'quadratic', 'linear']

# Implementing spatially varying length scale
def len_scale_func_linear(data_point_value):

    a = len_scale_params[0][0]
    b = len_scale_params[1][1]

    value =  a * data_point_value + b
    if value == 0:
        value = 1e-6

    return value


def len_scale_func_quad(data_point_value):

    a = len_scale_params[1][0]
    b = len_scale_params[1][1]
    c = len_scale_params[1][2]

    value =  a * (data_point_value ** 2 )+ b * data_point_value + c
    if value == 0:
        value = 1e-6
    return value


def var_kernel(data_point1, data_point2, signal_variance):
    kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
    for i in np.arange(len(data_point1)):
        for j in np.arange(len(data_point2)):
            len_scale_vectors = []
            for d in np.arange(len(data_point1[i,:])):

                if len_scale_func_type[d] == 'linear':
                    len_scale_vector_datapoint1 = len_scale_func_linear(data_point1[i, :][d])
                    len_scale_vector_datapoint2 = len_scale_func_linear(data_point1[j, :][d])

                elif len_scale_func_type[d] == 'quadratic':
                    len_scale_vector_datapoint1 = len_scale_func_quad(data_point1[i, :][d])
                    len_scale_vector_datapoint2 = len_scale_func_quad(data_point1[j, :][d])

                len_scale_vectors.append([len_scale_vector_datapoint1, len_scale_vector_datapoint2])

            difference = data_point1[i, :] - data_point2[j, :]
            total_product = 1
            total_sum = 0

            for k in np.arange(number_of_dimensions):
                denominator = len_scale_vectors[k][0] ** 2 + len_scale_vectors[k][1] ** 2
                total_product *= (2 * len_scale_vectors[k][0] * len_scale_vectors[k][1]) / denominator
                total_sum += 1 / denominator

            squared_diff = np.dot(difference, difference.T)
            each_kernel_val = (signal_variance ** 2) * np.sqrt(total_product) * (np.exp((-1) * squared_diff * total_sum))
            kernel_mat[i, j] = each_kernel_val

    return kernel_mat


number_of_dimensions = 3
number_of_observed_samples = 3
random_points = []
X = []
bounds = [[0, 1] for nd in np.arange(number_of_dimensions)]
# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                   number_of_observed_samples).reshape(1,
                                                                                       number_of_observed_samples)
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

# X = np.array([[1,2,1], [2,3,1], [3,1,1]])

print(X)
print(var_kernel(X, X, 1))
exit(0)

#DropWave
# x = np.linspace(-6,6,100)
# y = - (1+np.cos(12 * np.sqrt(x*x)))/(0.5 * x*x + 2)

#Ackley function
# x = np.linspace(-40,40,1000)
# y = -1 * (- 20 * np.exp(-0.2 * np.sqrt(x*x) ) - (np.exp(np.cos(2 * np.pi * x  ))) + 20 + np.exp(1))

#Alpine01 function
# x = np.linspace(-40,40,1000)
# y = abs(x * np.sin(x) + 0.1 * x)

#Alpine02 function
# x = np.linspace(0,10,1000)
# y = np.sqrt(x) * np.sin(x)

#Bird fn
# x = np.linspace(-10,10,1000)
# y = np.sin(x) * np.exp((1-np.cos(0))*(1- np.cos(0))) + np.cos(0) * np.exp((1-np.sin(x))*(1- np.sin(x))) + x * x

# Gramacy and Lee function
# x = np.linspace(0.5,3,1000)
# y = -1 * ((((np.sin(10 * np.pi * x))/(2*(x)))) +(x-1) ** 4)

#Benchmark Function
x = np.linspace(0,10, 1000)
y=  (np.exp(-x) * np.sin(2 * np.pi * x))

#Griewank Function
x = np.linspace(-600,600, 1000)
y=  x **2 /4000 - np.cos(x) +1

plt.plot (x,y)
plt.show()
exit(0)



#Normalizations

def fun(x):
    w = 1 +((x-1)/4)
    return  (np.sin(np.pi * w))**2 + ((w-1)**2)*(1+ (np.sin(2 * np.pi * w))**2)


x = np.linspace(-10, 10, 1000)


x =np.linspace (-10,10,10)
y = np.sin(x)
y = -fun(x)
# y = x * x +2
# plt.plot(x,y, "g")

xmax = x.max()
xmin = x.min()
xnew = (x - xmin) / (xmax - xmin)

# ymax = y.max()
# ymin = y.min()
# ynew = (y-ymin)/(ymax -ymin)

# ynew = ((y - y.mean())/ np.std(y))
ynew = y

# ynew = np.sin(x)

# ynew = np.sqrt(15/83) * (x*x +2)
# ynew = np.sqrt(15/83) * (xnew * xnew + 2)
# ynew = np.sin(x)
plt.plot(xnew, ynew, "r")

plt.show()
exit(0)


# Maximizers and Gradients
def multistart(fun, x0min, x0max, N, full_output=False, args=()):
    max = None
    func_max = -1 * float('inf')
    for i in range(N):
        # print("\n#########################")
        x0 = sp.random.uniform(x0min, x0max)
        res = sp.optimize.minimize(lambda x: -fun(x), x0, args
                                   # , jac= fun_grad
                                   , method='L-BFGS-B', bounds=[[x0min, x0max]]
                                   , options={
                                            'maxfun': 2000, 'maxiter': 2000
                                            # ,'disp':True
                                            }
                                   )

        if (res.success == False):
            print("Convergence failed, Skipping")
            continue

        val = -1 * res.fun
        # print("prev max:", func_max, "\tNew value: ", val, "\t at x=", res.x)
        if val > func_max:
            # print("New maximum found ")
            max = res
            func_max = val

    if max != None:
        print("max obtained is ", func_max
              # , " at ", max.x
              )

    return max


def fun(x):
    return (np.exp(-x) * np.sin(8 * np.pi * x)) + 1
    # return np.sin(x)
    # return - np.cos(x) + 0.01 * x ** 2 + 1
    # return ((np.exp(-x) * np.sin(3 * np.pi * x)) + 0.3)
    # return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)
    # return ( (np.sin(10* np.pi* x)/(2*x))+(x-1)**4)
    # w = 1 +((x-1)/4)
    # return  (np.sin(np.pi * w))**2 + ((w-1)**2)*(1+ (np.sin(2 * np.pi * w))**2)
    # return -np.sin(x) *(np.sin((x**2)/np.pi))**20


def fun_grad(x):
    return -1*(np.exp(-x) * (8 * np.pi * np.cos(8 * np.pi * x) - np.sin(8 * np.pi * x)))
    # return -1* np.cos(x)
    # ((np.exp(-x) * np.sin(3 * np.pi * x)) + 0.3)
    # return  (np.exp(-x) * ( 3*np.pi*np.cos(3 * np.pi * x) - np.sin(3 * np.pi * x)) )
    # (np.exp(-x) * np.sin(8 * np.pi * x)) + 1
    # return -2 *( np.exp(-(x - 2) ** 2) * (x - 2) + np.exp(-(x - 6) ** 2 / 10) * ((x - 6)/10) + x / (x ** 2 + 1))



con = []
for index in range(10):
    res_multi = multistart(fun, 0, 5, 10)
    if res_multi != None:
        con.append( str(res_multi.x[0]))


for each in con:
    print(each)

# Optimizer multi start

def multistart2(fun, x0min, x0max, N, full_output=False, args=(),
                method=None, jac=None, hess=None, hessp=None, bounds=None,
                constraints=(), tol=None, callback=None, options=None):
    res_list = sp.empty(N, dtype=object)
    for i in range(N):
        x0 = sp.random.uniform(x0min, x0max)
        res = sp.optimize.minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        res_list[i] = res

    sort_res_list = res_list[sp.argsort([res.fun for res in res_list])]
    if full_output:
        return sort_res_list[0], sort_res_list
    else:
        return sort_res_list[0]


# Normalizing

def normalize(x, y):
    xmax = x.max()
    xmin = x.min()
    xnew = (x - xmin) / (xmax - xmin)
    ymin = y.min()
    ymax = y.max()
    ynew = (y - ymin) / (ymax - ymin)
    return xnew, ynew


x = np.linspace(0, 10, 1000)
plt.plot(x, fun(x))
y1 = fun(x)
x1, y1 = normalize(x, y1)
plt.plot(x1, y1)
# print ( ynew)
plt.show()
exit(0)

# Frequency Counters
import collections

a = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5]
counter = collections.Counter(a)
print(counter)
exit(0)

# Array modifications
y = [10, 20, 30, 40, 5, 15, 50, 20]
true_max = 60
init_regret = []
for index in np.arange(len(y)):
    init_regret.append(true_max - max(y[:index + 1]))
print(init_regret)

regret = np.array([0.1, 0.2])
k = np.append(init_regret, regret)
print(k)
exit(0)


# Returning multiple values
def myfun(a, b):
    print(a, b)
    return a, b


# y = myfun(10,20)
# print(y[0])
a = np.array([10, 20, 30, 40, 50, 60])
print(a[:2])

b = [10, 20, 30, 40]

c = np.array(b)
print(type(b))

exit(0)

# Array manipulations
a = np.array([])
a = np.append(a, 10)
c = 3 / 5;
print(c)
print(a)

exit(0)


# Implementing spatially varying length scale
def len_scale_func(data_point):
    a = 0;
    b = 0;

    len_scale_weights = np.zeros(data_point.shape)
    len_scale_values = []
    for dim_count in np.arange(len(data_point)):
        len_scale_weights[dim_count] = 0.5
        len_scale_values.append(np.dot(len_scale_weights.T, data_point))
        print(len_scale_values)
        len_scale_weights[dim_count] = 0
    return len_scale_values


def sq_exp_kernel(data_point1, data_point2, signal_variance):
    kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
    for i in np.arange(len(data_point1)):
        for j in np.arange(len(data_point2)):

            len_scale_vector_datapoint1 = len_scale_func(data_point1[i, :])
            len_scale_vector_datapoint2 = len_scale_func(data_point2[j, :])
            difference = ((data_point1[i, :] - data_point2[j, :]))

            total_product = 1
            total_sum = 0

            for k in np.arange(len(len_scale_vector_datapoint1)):
                denominator = len_scale_vector_datapoint1[k] ** 2 + len_scale_vector_datapoint2[k] ** 2
                total_product *= (2 * len_scale_vector_datapoint1[k] * len_scale_vector_datapoint2[k]) / denominator
                total_sum += 1 / denominator

            squared_diff = np.dot(difference, difference.T)
            each_kernel_val = (signal_variance ** 2) * np.sqrt(total_product) * (
                np.exp((-1) * squared_diff * total_sum))
            kernel_mat[i, j] = each_kernel_val

    return kernel_mat


number_of_dimensions = 3
number_of_observed_samples = 3
random_points = []
X = []
bounds = [[0, 1] for nd in np.arange(number_of_dimensions)]
# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                   number_of_observed_samples).reshape(1,
                                                                                       number_of_observed_samples)
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

# X = np.array([[1,2,1], [2,3,1], [3,1,1]])

print(X)
print(sq_exp_kernel(X, X, 1))
exit(0)


# modified way of implementing the squared exponential kernel
def sq_exp_kernel(data_point1, data_point2, char_len_scale, signal_variance):
    print(data_point1)
    char_len_scale = char_len_scale ** 2
    sq_dia_len = np.diag(char_len_scale)
    inv_sq_dia_len = np.linalg.pinv(sq_dia_len)
    print(inv_sq_dia_len)
    kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
    for i in np.arange(len(data_point1)):
        for j in np.arange(len(data_point2)):
            difference = ((data_point1[i, :] - data_point2[j, :]))
            product1 = np.dot(difference, inv_sq_dia_len)
            final_product = np.dot(product1, difference.T)
            each_kernel_val = signal_variance * (np.exp((-1 / 2.0) * final_product))
            kernel_mat[i, j] = each_kernel_val
    return kernel_mat


random_points = []
X = []
number_of_dimensions = 2
number_of_observed_samples = 5
bounds = [[0, 1], [10, 11]]

# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1], \
                                                   number_of_observed_samples).reshape(1, number_of_observed_samples)
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

# X = np.array([[1,1],[2,3],[1,1]]).reshape(-1,2)
X = np.array([[0.78384805, 0.78592446, 0.10494611, 0.3704902, 0.12773128, 0.76679997],
              [0.1160734, 0.89851246, 0.95711753, 0.49591683, 0.35312216, 0.94562597],
              [0.78703226, 0.69398919, 0.63503412, 0.0478438, 0.63185067, 0.02005047],
              [0.09468655, 0.95879682, 0.57056379, 0.78355148, 0.03670761, 0.21144576],
              [0.33898591, 0.13382438, 0.69899874, 0.4687739, 0.80211813, 0.01479639],
              [0.64657258, 0.50300766, 0.1655368, 0.97593827, 0.19528553, 0.3692436],
              [0.14909383, 0.36605532, 0.34926196, 0.50828534, 0.1554955, 0.47065956]]).reshape(-1, 6)

# print(X)

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = np.append(a, b)
len_scale_bounds = [0.1, 1]
number_of_dimensions = 3
print([len_scale_bounds for nd in np.arange(number_of_dimensions + 1)])

v_char_len_scale = [1, 1, 1, 2, 1, 1]
print(v_char_len_scale ** 2)
v_char_len_scale = np.array([1 for n in np.arange(6)])
print(v_char_len_scale)

signal_variance = 1

print("printing ")


# print(sq_exp_kernel(X, X, v_char_len_scale, signal_variance))
# print(osq_exp_kernel(X, X, char_len_scale))


def ard_sq_exp_kernel(data_point1, data_point2, char_len_scale):
    # k(x1,x2) = sig_squared*exp{(-1/2*(charac_length_scale))((euclidean_distance(x1,x2)**2))}
    # Parameter sig_squared = 1
    # distances((a1,a2),(b1,b2)) = sqrt(a1**2 + a2**2 + b1**2 + b2**2 + 2(a1*b1 + a2*b2))
    # total_squared_distances = np.sum(np.square(data_point1), 1).reshape(-1, 1) + np.sum(np.square(data_point2), 1) \
    #                           - 2 * np.dot(data_point1, data_point2.T)
    a = np.array([[1, 2, 5]])
    b = np.array([[1, 2, 3]])
    print(np.linalg.norm(a - b))
    char_len_scale = char_len_scale ** 2
    print("c", char_len_scale)
    sq_dia_len = np.diag(char_len_scale)
    print("s", sq_dia_len)
    inv_sq_dia_len = np.linalg.pinv(sq_dia_len)
    print("i", inv_sq_dia_len)
    factor1 = np.dot(difference, inv_sq_dia_len)
    print("f", factor1)
    product = np.dot(factor1, difference.T)
    kernel_val = 1 * (np.exp((-1 / 2.0) * product))
    return kernel_val


exit(0)

# matrix square inverse
char_len_scale = np.array([1, 2, 3])
char_len_scale = char_len_scale ** 2

sq_dia_len = np.diag(char_len_scale)
inv_sq_dia_len = np.linalg.pinv(sq_dia_len)
print(inv_sq_dia_len)
exit(0)

# Hartmann 6D functions
x = np.matrix([[0.30319131, 0.96140171, 1., 0., 0., 1.]])
alpha = np.array([1.0, 1.2, 3.0, 3.2])
A_array = [[10, 3, 17, 3.50, 1.7, 8], [0.05, 10, 17, 0.10, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
A = np.matrix(A_array)
P_array = [[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047,
                                                                                       6650], [4047, 8828, 8732, 5743, 1091, 381]]
P = np.matrix(P_array)
P = P * 1e-4
sum = 0
for i in np.arange(0, 4):
    alpha_value = alpha[i]
    inner_sum = 0
    for j in np.arange(0, 6):
        inner_sum += A.item(i, j) * ((x[:, j] - P.item(i, j)) ** 2)
    sum += alpha_value * np.exp(-1 * inner_sum)
    # extra -1 is because we are finding maxima instead of the minima f(-x)
value = (-1 * -1 * sum).reshape(-1, 1)

print(value)
exit(0)

# matrix square inverse
char_len_scale = np.array([1, 2, 3])
char_len_scale = char_len_scale ** 2

sq_dia_len = np.diag(char_len_scale)
inv_sq_dia_len = np.linalg.pinv(sq_dia_len)
print(inv_sq_dia_len)

exit(0)

# Writing log files
import sys


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


f = open('out.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)
print("test")  # This will go to stdout and the file out.txt

# use the original
sys.stdout = original
print("This won't appear on file")  # Only on stdout
f.close()
exit(0)

# matrix computations
v = np.array([1, 2, 3]).reshape(-1, 1)
print(np.dot(v.T, np.eye(3)))
exit(0)

# Time calculations
now = datetime.datetime.now()
print("Current date and time : ")
acq_fun_list = ['ei', 'pi', 'ucb', 'rs']
if ('ei2' in acq_fun_list):
    print(now.strftime("%H%M%S_%Y%m%d"))
exit(0)

# vector differences
a = np.array([[1, 2], [3, 4]]).reshape(-1, 2)
b = np.array([[3, 3]]).reshape(-1, 1)
print((a - b) ** 2)
exit(0)

# generating a random number of given dimensions
n = 2
bounds = [[10, 15], [20, 25], [35, 40]]
array = np.array([])
for dim in np.arange(n):
    value = np.random.uniform(bounds[dim][0], bounds[dim][1], 1).reshape(1, 1)
    array = np.append(array, value)
    print()
print(array)
exit(0)

# matrix multiply with list
z = np.array([1, 2])
zz = np.array([[1, 2, 1], [3, 1, 1]])
print(np.dot(z, zz))
exit(0)

# plotting 3d graph
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-100, 100, 0.1)
Y = np.arange(-100, 100, 0.1)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
# Z = np.sin(R)

Z = X ** 2 + Y ** 2

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# # Customize the z axis.
# ax.set_zlim(0, 10000)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

exit(0)

# datapoints difference for se kernel
dp1 = np.matrix([[1, 2], [6, 4]])
dp2 = [[4, 5]]

print(dp1.shape, type(dp1))
print(np.square(dp1))
print()
print(np.sum(np.square(dp1), 1).reshape(-1, 1))
print(np.zeros(2))
distances = np.sum(dp1 ** 2, 1).reshape(-1, 1) + np.sum(dp2 ** 2, 1) - 2 * np.dot(dp1, dp2.T)
print(distances)
exit(0)

# array appending and continue
temp = []
for i in arange(10):
    if (i % 2 == 0):
        continue
    print(i)
temp.append(22)
temp.append(33)
print(temp)
exit(0)

# plotting the legend
a = np.arange(0, 10, 1)
c = (a * a)
err = [10, 5, 10, 5, 20, 30, 5, 10, 40, 20]
plt.plot(a, c, color='blue', label='Model length')
# plt.errorbar(a,c, yerr= err)
plt.show()
exit(0)

# plotting the graph with legends
a = b = np.arange(0, 10, 1)
c = np.exp(a)
d = c[::-1]
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c + d, 'k', label='Total message length')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
plt.show()
exit(0)

# reshaping the array
z = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print(z.reshape(1, -1))
exit(0)

# plotting the legend with mpatches
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(color='red', label='The red data')
plt.legend(handles=[red_patch])
plt.show()
exit(0)

number_of_iterations = 10
iteration_axes_values = [[i + 1] for i in arange(number_of_iterations)]
print(iteration_axes_values)
exit(0)

# total_ucb_regret = np.array([[]], ndmin=2)
init_matrix = []

print(init_matrix)

a = np.array([1, 2, 3, 4, 5, 6])
a = a.reshape(1, 6)

b = np.array([3, 6, 7, 8, 9, 6])
b = b.reshape(1, 6)

c = np.array([2, 6, 3, 4, 5, 2])
c = c.reshape(1, 6)

d = np.array([6, 5, 5, 2, 5, 5])
d = d.reshape(1, 6)

init_matrix.append(a)
init_matrix.append(b)
# init_matrix.append(c)
# init_matrix.append(d)


stacked = np.vstack(init_matrix)
print(np.std(stacked, axis=0))
print("aaaa", np.mean(stacked, axis=None))

exit(0)

arr = np.empty((0, 6), int)
print(arr)
arr = np.append(arr, np.array([[1, 2, 3, 4, 5, 6]]), axis=0)
arr = np.append(arr, np.array([[1, 2, 3, 4, 5, 6]]), axis=0)
print(arr)
exit(0)

#
#
# otal_ucb_regret = np.append(total_ucb_regret,a, axis=0)
# print(total_ucb_regret)

b = np.array([7, 8, 9, 7, 1, 6])
b = b.reshape(len(b), 1)
# total_ucb_regret = total_ucb_regret.reshape(6,1)
total_ucb_regret = np.hstack([total_ucb_regret, b])
# total_ucb_regret = np.hstack(total_ucb_regret, a)


input_array = np.array([1, 2, 3]).reshape(3, 1)
new_row = np.array([4, 5, 6]).reshape(3, 1)

new_array = np.hstack([input_array, new_row])
print(new_array)

# exit(0)


a = [[-4],
     [-3],
     [-2],
     [-1],
     [0]]

print(type(a))
b = [4]
aa = np.append(a, [[5]], axis=0)

print(aa)

exit(0)


class A:

    # def __init__(self, kappa = 1):
    #     self.kappa = kappa

    def f(self, x):
        return -1 * (np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1))
        # return -1* np.sin(x)


if __name__ == "__main__":
    a = A()

    boundaries = [(-3, 5.0)]

    starting_points = np.random.uniform(boundaries[0][0], boundaries[0][1], 20)

    best_obs = 0
    x = 0
    for starting_point in starting_points:
        print("starting point ", starting_point)
        max_x = opt.minimize(a.f, x0=starting_point, method='L-BFGS-B', bounds=boundaries)

        value = -1 * a.f(max_x['x'])

        if (value > best_obs):
            print("New best value found ", max_x['x'])
            best_obs = value
            x = max_x['x']
            print("start: ", starting_point, "observed: ", best_obs, " ******* ")

    print("Best is ", value, "at ", x)

    x = np.linspace(-4, 8, 100)
    y = -1 * a.f(x)

    plt.figure('4')
    plt.clf()
    plt.plot(x, y)
    plt.axis([-6, 8, -3, 3])
    plt.title('acq function')
    # plt.savefig('acq', bbox_inches='tight')
    plt.show()

#
# class Acq_Maximizer(object):
#
#     def func(self, params):
#         x, y = params
#         return np.sin(x)
#
#     def maximizer(self, x, y):
#         # x,y = params
#         params = [x,y]
#         print (x,y)
#         value = (self.func(params)) ** 2
#         return -1 * value
#
#     def start_max(self):
#         return
#
# if __name__ == "__main__":
#     acq_max = Acq_Maximizer()
#
#     start1 = 1
#     start2 = 2
#     parameters = [start1, start2]
#     boundaries = [(-2, 10), (-2, 10)]
#     result = opt.minimize(acq_max.maximizer, x0=parameters, bounds=boundaries)
#     print(result)


# **************************


# for i in range(1,10):

# # z = lambda x, y: -f(x,y)
# max_x = opt.fmin_l_bfgs_b( lambda x: -f1(x) , 1, bounds=[(-2,10.0)],approx_grad=True)
# print(max_x)

#
#
#
#
# def functionyouwanttofit(x,y):
#     return np.array([x+y, x-y]) # baby test here but put what you want
#
# def calc_chi2(parameters):
#     x,y = parameters
#     data = np.array([100,250,300,500])
#     chi2 = sum( (data-functionyouwanttofit(x,y))**2 )
#     return -1*chi2
#
# # # baby example for init, min & max values
# x_init = 0
# x_min = -1
# x_max = 10
# y_init = 1
# y_min = -2
# y_max = 9
# # z_init = 2
# # z_min = 0
# # z_max = 1000
# # t_init = 10
# # t_min = 1
# # t_max = 100
# # u_init = 10
# # u_min = 1
# # u_max = 100
# parameters = [x_init,y_init]
# bounds = [[x_min,x_max],[y_min,y_max]]
# result = opt.minimize(calc_chi2,parameters,bounds=bounds)
# print(result)
#
#

# for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
#
#         res = minimize(fun=acquisition_func,
#                        x0=starting_point.reshape(1, -1),
#                        bounds=bounds,
#                        method='L-BFGS-B',
#                        args=(gaussian_process, evaluated_loss, greater_is_better, n_params))
#
#         if res.fun < best_acquisition_value:
#             best_acquisition_value = res.fun
#             best_x = res.x
#
#     return best_x
