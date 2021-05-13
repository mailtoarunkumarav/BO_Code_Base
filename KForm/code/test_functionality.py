import numpy as np


a= [2]
print(np.square(a))
exit(0)

X1 = np.array([1,2,3])
X2 = np.array([4,5,6])

print(np.linalg.norm(X1-X2))

X = np.array([[1,2,3],[4,5,6], [10,20,30], [21,22,13]])
print(X[:2])
exit(0)

a = [0.2 for d in np.arange(2)]
print(a)
exit(0)

#Squaring and summing
a = [1,2]
squ = np.square(a)
sum = np.sum(squ)
print(squ,sum)
exit(0)


# Testing slicing

a = [10,20,30,40, 50 , 60]
b = a[:2]
print(b, "dddd", a[2:])

# Testing kernel
number_of_dimensions = 2
number_of_observed_samples = 10
bounds = [[0,2],[2,3]]
random_points = []
X = []

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

print(X.shape, X)
norms = np.linalg.norm(X, axis=1)


rand_vector1 = [3]
rand_vector2 = [5,5]

# dp1 = np.array([[1, 2],[3,4]])
# dp2 = np.array([[4, 5], [10,20], [2, 3], [5,8]])
data_point1 = np.array([[1,2], [2,3]])
data_point2 = np.array([[4,5], [10,20], [2,3], [5,8]])

kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
for i in np.arange(len(data_point1)):
    for j in np.arange(len(data_point2)):
        difference = ((data_point1[i, :] - data_point2[j, :]))
        each_kernel_val = np.multiply(data_point1[i, :], data_point2[j, :])
        rand_vector_product = np.multiply(rand_vector1, rand_vector2)
        total_product = np.multiply(each_kernel_val, rand_vector_product)
        each_kernel_val =  np.sum(total_product)
        kernel_mat[i, j] = each_kernel_val
print(kernel_mat)




