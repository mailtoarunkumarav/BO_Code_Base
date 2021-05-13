import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as scy_lin
import sys, getopt
import datetime


argv = sys.argv[1:]
dataset = ""
subspaces =""
iterations = ""

cmd_inputs = {}

try:
    opts, args = getopt.getopt(argv, "d:s:t:", ["dataset=", "subspace=", "iterations="])
except getopt.GetoptError:
    print('test.py -d <dataset> -s <number_of_subspaces> -t <number_of_iterations>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-d", "--dataset"):
        cmd_inputs["dataset"] = arg
    elif opt in ("-s", "--subspace"):
        cmd_inputs["subspace"] = arg
    elif opt in ("-t", "--iterations"):
        cmd_inputs["iterations"] = arg
    else:
        print('test.py -d <dataset> -s <number_of_subspaces> -t <number_of_iterations>')
        sys.exit()



exit(0)

### Printing to standard output and
import os
# sys.path.append("..")
# from HelperUtility.PrintHelper import PrintHelper

class PrintHelper:

    def __init__(self, directory):
        original_stdout = sys.stdout
        timenow = datetime.datetime.now()
        stamp = timenow.strftime("%H%M%S_%d%m%Y")
        PrintHelper.console_output = open(directory+'/console_output_' + str(stamp) + '.txt', 'w')

    @staticmethod
    def printme(*args):

        if type(args[0]) is bool:
            if args[0]:
                for each1 in args[1:]:
                    print(each1, end=' ')
                print("")

            for each2 in args[1:]:
                PrintHelper.console_output.write(each2 + " ")
            PrintHelper.console_output.write("\n")

priority1 = True
priority2 = False
priority3 = False
priority4 = True
PrintHelper(os.getcwd())
# With old type of priority
PrintHelper.printme(priority1, "hello", "hello2")
PrintHelper.printme(priority2, "hello3", "hello4")
PrintHelper.printme(priority3, "hello5")
PrintHelper.printme('')
PrintHelper.printme('abs')
PrintHelper.printme(priority4, 'abc')
exit(0)





# Gaussian Mixtures
x = np.linspace(0, 15, 1000)
y = np.array([])
for each_x in x:
    if each_x <= 5:
        sig = 0.4
        mean = 2.5

    elif each_x > 5 and each_x <= 10:
        sig = 0.7
        mean= 7.5

    elif each_x > 10 and each_x <= 15:
        sig = 0.6
        mean= 12.5

    val = 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((each_x - mean)/sig)*(each_x - mean)/sig)
    y = np.append(y, val)
y = y.reshape(-1, 1)
plt.plot(x, y)
plt.show()
exit(0)


# Sinc mixtures
x = np.linspace(-15, 15, 1000)
y = np.sinc(x-10) + np.sinc(x) + np.sinc(x+10)
plt.plot(x, y)
plt.show()
exit(0)

#Chirpwave function modified

x = np.linspace(0, 20, 1000)
y = np.array([])
f = 1
for each_x in x:
    if each_x < 8:
        f = 0.35
    elif each_x > 8 and each_x <= 15:
        f = 1.25
    elif each_x > 15 and each_x <= 20:
        f = 0.35
    val = np.sin(2 * np.pi * f * each_x)
    y = np.append(y, val)
y = y.reshape(-1, 1)

plt.plot(x, y)

x_obs = np.linspace(0, 7.5, 40)
x_obs = np.append(x_obs,np.linspace(16, 20, 30))
# x_obs = np.random.uniform(0, 8, 12)
# x_obs = np.append(x_obs,np.random.uniform(8,15,13))

reshaped = x_obs.reshape(-1, 1)
print(np.sort(reshaped, axis=0))

# X = np.array([
# [ 0.320835626],
# [ 1.40835626],
# [ 2.45835626],
# [ 3.20835626],
# [ 4.20835626],
#  [ 5.60195798],
#  [ 6.42273538],
#  [ 7.21350942],
#  [ 7.32440451],
#  [ 7.90691315],
#  [ 8.32949304],
#  [ 8.67509187],
#  [ 8.90218965],
#  [ 9.55831041],
#  [10.43239387],
#  [10.64328088],
#  [10.94865737],
#  [11.6755274 ],
#  [12.354817602],
#  [12.93994599],
#  [13.20932905],
#  [14.22347481],
# [14.3232872],
#  [14.5357832 ],
#  [15.4473578 ],
#  [15.973578],
#  [17.34861657],
#  [18.16232872],
#  [18.95330988],
# [19.95330988]])

# x_obs = X

y_obs = np.array([])
for each_x_obs in x_obs:
    if each_x_obs< 8:
        f = 0.35
    elif each_x_obs > 8 and each_x_obs<= 15:
        f = 1.25
    elif each_x_obs > 15 and each_x_obs<= 20:
        f = 0.35
    val = np.sin(2 * np.pi * f * each_x_obs)
    y_obs = np.append(y_obs, val)
y_obs = y_obs.reshape(-1, 1)
plt.plot(x_obs,y_obs,"r+")
plt.show()
exit(0)



def polynomial_kernel( datapoint1, datapoint2, datapoint3, datapoint4):
    hyper_lambda = 0.5
    num = 1 - hyper_lambda
    prod1 = np.dot(datapoint1, datapoint2.T)
    prod2 = np.dot(datapoint3, datapoint4.T)
    val = ((1 + prod1 + prod2 + np.dot(prod1, prod2)) * (1 + prod1 + prod2 + np.dot(prod1, prod2)))
    # val = val / (((1 + number_of_dimensions) * (1 + number_of_dimensions)) * (
    #             (1 + number_of_dimensions) * (1 + number_of_dimensions)))
    den = 1 - (hyper_lambda * val)
    kernel_val = num / den
    return kernel_val


# Kappa recalculation
def gaussian_harmonic_hyperkernel(datapoint1, datapoint2, datapoint3, datapoint4):
    lambda_hyperk = 0.5
    num = 1 - lambda_hyperk
    sigma_variance = 1
    difference1 = datapoint1-datapoint2
    difference2 = datapoint3-datapoint4
    den = 1 - lambda_hyperk * (np.exp(-1 * (sigma_variance ** 2) * ((np.dot(difference1, difference1.T)) +
                                                                    (np.dot(difference2, difference2.T)))))
    kernel_val = num / den


    return kernel_val


def compute_kernel_with_hyperkernel(x1, x2, x3, x4):
    # stri = str(str(x1) + "-" + str(x2) + "-" + str(x3) + "-" + str(x4))
    # return stri
    # updated implmntn

    hyperkernel = 'gaussian_harmonic'
    if hyperkernel == 'gaussian_harmonic':
        covariance = gaussian_harmonic_hyperkernel(x1, x2, x3, x4)

    elif hyperkernel == 'polynomial':
        covariance = polynomial_kernel(x1, x2, x3, x4)

    return covariance


def compute_covariance_in_X2(X):
    total_possible_sample_pairs = len(X) * len(X)
    # kappa_xtil_xtil = np.empty((total_possible_sample_pairs,total_possible_sample_pairs), dtype= str)
    # kappa_xtil_xtil = [["" for x in range(total_possible_sample_pairs)] for i in range(total_possible_sample_pairs)]
    #updated implmtn
    kappa_xtil_xtil = np.zeros(shape=(total_possible_sample_pairs, total_possible_sample_pairs))

    mat_index1 = 0
    mat_index2 = 0

    for index1 in range(len(X)):
        for index2 in range(len(X)):
            for index3 in range(len(X)):
                for index4 in range(len(X)):

                    kappa_xtil_xtil[mat_index1][mat_index2] = compute_kernel_with_hyperkernel(X[index1], X[index2], X[index3], X[index4])
                    print("["+str(mat_index1)+"]["+str(mat_index2)+"]\t"+str(X[index1][0]) +"_"+ str(X[index2][0]) +"_"+  str(X[index3][0])
                          +"_"+str(X[index4][0]))
                    mat_index2 = (mat_index2 + 1) % (len(X) * len(X))
            mat_index1 = (mat_index1 + 1) % (len(X) * len(X))

    # for kap_i in range(0, total_possible_sample_pairs):
    #     for kap_j in range(0, total_possible_sample_pairs):
    #         index1 = int(kap_i / len(X))
    #         index2 = int(kap_j / len(X))
    #         index3 = kap_i % len(X)
    #         index4 = kap_j % len(X)
    #
    #         # kappa_xtil_xtil[kap_i][kap_j] = str(stri)
    #         kappa_xtil_xtil[kap_i][kap_j] = compute_kernel_with_hyperkernel(X[index1], X[index2], X[index3], X[index4])

    return kappa_xtil_xtil


number_of_samples_in_X = 3
X_space_min = 0
X_space_max = 2
number_of_observed_samples = 4
number_of_dimensions = 1
bounds = [[1, 4], [1, 4]]

X = []

# X = np.linspace(X_space_min, X_space_max, number_of_samples_in_X).reshape(number_of_samples_in_X, number_of_dimensions)

# updated implmntn
random_points = []
# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    # random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1], number_of_observed_samples).reshape(1,
    #                                                                                     number_of_observed_samples)

    random_data_point_each_dim = np.linspace(X_space_min, X_space_max, number_of_samples_in_X).reshape(1, number_of_samples_in_X)
    random_points.append(random_data_point_each_dim)

# Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
random_points = np.vstack(random_points)

# Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
for sample_num in np.arange(number_of_samples_in_X):
    array = []
    for dim_count in np.arange(number_of_dimensions):
        array.append(random_points[dim_count, sample_num])
    X.append(array)
X = np.vstack(X)

print(X)

kernel_matrix_hk = None
kernel_matrix_hk = compute_covariance_in_X2(X)
print(kernel_matrix_hk)
exit(0)
# Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
cholesky_kernel_matrix_hk = np.linalg.cholesky(kernel_matrix_hk + 1e-6 * np.eye(len(X) * len(X)))

# Sample 3 standard normals for each of the unseen data points
standard_normals = np.random.normal(size=(len(X) * len(X), 2))

# multiply them by the square root of the covariance matrix L
kernel_samples = np.dot(cholesky_kernel_matrix_hk, standard_normals)
exit(0)






# Fixing the Kappa matrix issue

X = np.array(["a", "b"])

y = X[0]+X[1]
print(y)

grid_pts = len(X)
kappa_matrix = np.chararray((len(X)*len(X), len(X)*len(X)), itemsize=5)
mat_index1 = 0
mat_index2 = 0

for index1 in range(len(X)):
    for index2 in range(len(X)):
        for index3 in range(len(X)):
            for index4 in range(len(X)):
                val = X[index1]+X[index2]+X[index3]+X[index4]
                print(val)
                kappa_matrix[mat_index1][mat_index2] = val
                mat_index2 = (mat_index2 + 1) % (len(X) * 2)
        mat_index1 = (mat_index1 + 1) % (len(X) * 2)

print(kappa_matrix)
exit(0)

# Plotting grid_vs_MLE

x = np.array([5, 10, 15, 20, 30, 50, 100])
y = np.array([12.65, 17.60, 17.70, 17.76, 17.81, 17.89, 18.34])

plt.plot(x,y)
plt.xlabel("Grid size")
plt.ylabel("Log marginal likelihood")
# plt.xticks([5,10,15,20,30,50,100])
plt.plot()
plt.show()
exit(0)





# Vectorisation and Matrix conversion
vector = np.random.normal(size=(100, 1))
print(vector.shape)

vector2 = vector.reshape(10, 10)
print(vector.shape)
print(vector)





# incomplete cholesky decomposition
def incomplete_cholesky_decomposition(k):

    mat_length = np.shape(k)[0]
    print(mat_length)
    for c in range(mat_length):
        k[c][c] = np.sqrt(k[c][c])

        for i in range(c+1, mat_length):
            if k[i][c] != 0:
                k[i][c] = k[i][c]/k[c][c]

        for j in range(c+1, mat_length):
            for p in range(j, mat_length):
                if k[p][j] != 0:
                    k[p][j] = k[p][j] - k[p][c] * k[j][c]

    for m in range(mat_length):
        for n in range(m+1, mat_length):
            k[m][n] = 0

    return k


signal_variance = 1
char_len_scale = 0.3
X = np.linspace(0, 10, 3).reshape(-1,1)
data_point1 = X
data_point2 = X

kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
for i in np.arange(len(data_point1)):
    for j in np.arange(len(data_point2)):
        difference = (data_point1[i, :] - data_point2[j, :])
        l2_difference_sq = np.dot(difference, difference.T)
        each_kernel_val = (signal_variance ** 2) * (np.exp((-1 / (2 * char_len_scale ** 2)) * l2_difference_sq))
        kernel_mat[i, j] = each_kernel_val

eye =  1e-6 * np.eye(len(X))
L_x_x = np.linalg.cholesky(kernel_mat+eye)


kernel_test =  np.array([[1, 0.5, 0.75],
                        [0.5, 1 ,0.25],
                        [0.75, 0.25, 1]])
kernel_test = kernel_test.reshape(3,3)
print(kernel_test)
inc_L = incomplete_cholesky_decomposition(kernel_test)
print(inc_L)


print(L_x_x)
exit(0)


# Chirpwave function
# Merging the domains for damped oscillator
x = np.linspace(0,20,1000)
y = np.array([])
f = 1
for each_x in x:
    if each_x < 8:
        f=0.35
    elif each_x > 8 and each_x <=15:
        f=1.25
    elif each_x > 15 and each_x <=20:
        f=0.35
    val = np.sin(2 * np.pi * f * each_x)
    y= np.append(y, val)

print(y)
plt.xlabel("X")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
exit(0)



#Trial 2 working # Testing triangular wave function
X = np.linspace(0, 10,1000)
y_arr = (2*np.arcsin(np.sin(np.pi*X)))/(np.pi)
plt.xlabel("X")
plt.ylabel("f(x)")
plt.plot(X,y_arr)
plt.show()
exit(0)

#Trial 1 Failed # Testing triangular wave function
X = np.linspace(0,3,1000)
y_arr = []
val = 0

for each_x in X:
    for i in range(100):
        n = 2 * i + 1
        f = 2
        val = val + np.power(-i, n) * np.sin(2 * np.pi * f * each_x) *(1/(n*n))
    val = val * (8 / (np.pi * np.pi))
    y_arr.append(val)

plt.plot(X,y_arr)
plt.show()
exit(0)


# Merging the domains for damped oscillator
x = np.linspace(0,30,1000)
y = np.array([])
for each_x in x:
    if each_x < 10:
        # val = 2
        val = (np.exp(-each_x) * np.sin(1.5 * np.pi * each_x)) + 1
    elif each_x > 10 and each_x <20:
        val = 2
        each_x = each_x - 10
        val = (np.exp(-each_x) * np.sin(1.5 * np.pi * each_x)) + 1
    elif each_x > 20 and each_x <30:
        val = 2
        each_x = each_x - 20
        val = (np.exp(-each_x) * np.sin(1.5 * np.pi * each_x)) + 1
    y = np.append(y,val)

print(y)
plt.xlabel("X")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
exit(0)


# One line function for merged damped oscillator
# Not working
x = np.linspace(0,19.99,1000)
y = (math.floor(math.log(x,10))-1)*((np.exp(-x) * np.sin(1.5 * np.pi * x)) + 1)+ (math.floor(x%10))*((np.exp(-x) * np.sin(1.5 * np.pi * x))
                                                                                                  + 1)
plt.plot(x,y)
plt.show()
exit(0)


# Square wave calculations
# square wave function
x = np.linspace(0,4,100)
y = np.array([])

for each_x in x:
    each_y = np.sin(2 * np.pi * each_x)
    if each_y < 0:
        each_y = -1
    elif each_y > 0:
        each_y = 1
    else:
        each_y = 0
    y = np.append(y, each_y)

plt.plot(x,y)
plt.show()
exit(0)


# testing for eigen value devcomposition
mat = np.diag([1,4,9])
print(mat)
print(np.sqrt(mat))
print(math.floor(math.log(0.0127, 10)))

print(np.linspace(1,10,10))
exit(0)

# Testing symmetrizing the matrix

X = np.array([[0.94763226], [0.22654742],
              [0.59442014],
              [0.42830868],
              [0.76414069],
              [0.00286059],
              [0.35742368],
              [0.90969489],
              [0.45608099],
              [0.98180271]])
y = np.array([[0.39998281],
              [0.41574062],
              [0.39896644],
              [0.4040792],
              [0.39992303],
              [0.65600741],
              [0.41073034],
              [0.40002903],
              [0.40417766],
              [0.40002157]])

plt.plot(X,y)
plt.show()

exit(0)

# testing with np.append() matrix for the computation
b = np.array([]).reshape(-1,1)
a = np.array([[1],[2],[3]]).reshape(-1,1)
b = np.append(b, a, axis=0)
print(b)
e = np.array([[10],[20],[30]]).reshape(-1,1)
b = np.append(b, e, axis=0)
print(b)
exit(0)
# b = np.array([2,4])
# a = np.append(a,[b],axis =0)
# print(a)
c = np.array([5])
print(c)
a = np.append(a, c, axis=0)
print(a)


# Minimum calculations in an array
exit(0)
X= np.array([[10],[2],[3],[4],[1],[5]]).reshape(-1, 1)
print(X, X.argmin())
exit(0)


#Sanity checks for the mean computations
kappa = np.array([[1,         0.61269984, 0.6126998, 0.53628944],
 [0.61269984, 1,         0.53628944, 0.61269984],
 [0.61269984, 0.53628944, 1,         0.61269984],
 [0.53628944, 0.61269984, 0.61269984, 1        ]])
print(kappa)
alpha_diff = np.array([-0.02473023,  0.120585,    0.10650046,  0.13851545] )

value = np.exp((-1/(2*0.09))*np.sqrt(np.dot(alpha_diff,(np.dot(kappa, alpha_diff)))))
print(value)

exit(0)
# appending and array for multiple values of lambda

lambda_weights = np.array([[1,2,3],[4,5,6]])
print("lambda gp ", lambda_weights)
a = []
a.append(lambda_weights)
print(a)
a.append([1, 2, 3])
a = np.vstack(a)
print(a)

for i in range(len(lambda_weights)):
    print(i, lambda_weights[i])

exit(0)



# Gaussian harmonic kernel matrix generation code

def gaussian_harmonic_hyperkernel(datapoint1, datapoint2, datapoint3, datapoint4):
    lambda_hyperk = 0.5
    num = 1 - lambda_hyperk
    sigma_variance = 1
    difference1 = datapoint1-datapoint2
    difference2 = datapoint3-datapoint4
    den = 1 - lambda_hyperk * (np.exp(-1 * (sigma_variance ** 2) * ((np.dot(difference1, difference1.T)) +
                                                                    (np.dot(difference2, difference2.T)))))
    kernel_val = num / den
    return kernel_val


def compute_kernel_with_hyperkernel(x1, x2, x3, x4):
    # stri = str(str(x1) + "-" + str(x2) + "-" + str(x3) + "-" + str(x4))
    # return stri
    # updated implmntn

    hyperkernel = 'gaussian_harmonic'
    if hyperkernel == 'gaussian_harmonic':
        covariance = gaussian_harmonic_hyperkernel(x1, x2, x3, x4)

    return covariance


def compute_covariance_in_X2(X):
    total_possible_sample_pairs = len(X) * len(X)
    # kappa_xtil_xtil = np.empty((total_possible_sample_pairs,total_possible_sample_pairs), dtype= str)
    # kappa_xtil_xtil = [["" for x in range(total_possible_sample_pairs)] for i in range(total_possible_sample_pairs)]
    #updated implmtn
    kappa_xtil_xtil = np.zeros(shape=(total_possible_sample_pairs, total_possible_sample_pairs))

    for kap_i in range(0, total_possible_sample_pairs):
        for kap_j in range(0, total_possible_sample_pairs):
            index1 = int(kap_i / len(X))
            index2 = int(kap_j / len(X))
            index3 = kap_i % len(X)
            index4 = kap_j % len(X)

            # kappa_xtil_xtil[kap_i][kap_j] = str(stri)
            kappa_xtil_xtil[kap_i][kap_j] = compute_kernel_with_hyperkernel(X[index1], X[index2], X[index3], X[index4])

    return kappa_xtil_xtil


number_of_samples_in_X = 2
X_space_min = 1
X_space_max = 2
number_of_observed_samples = 4
number_of_dimensions = 1
bounds = [[1, 4], [1, 4]]

X = []

# X = np.linspace(X_space_min, X_space_max, number_of_samples_in_X).reshape(number_of_samples_in_X, number_of_dimensions)

# updated implmntn
random_points = []
# Generate specified (number of observed samples) random numbers for each dimension
for dim in np.arange(number_of_dimensions):
    # random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1], number_of_observed_samples).reshape(1,
    #                                                                                     number_of_observed_samples)

    random_data_point_each_dim = np.linspace(X_space_min, X_space_max, number_of_samples_in_X).reshape(1, number_of_samples_in_X)
    random_points.append(random_data_point_each_dim)

# Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
random_points = np.vstack(random_points)

# Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
for sample_num in np.arange(number_of_samples_in_X):
    array = []
    for dim_count in np.arange(number_of_dimensions):
        array.append(random_points[dim_count, sample_num])
    X.append(array)
X = np.vstack(X)

print(X)

kernel_matrix_hk = None
kernel_matrix_hk = compute_covariance_in_X2(X)
print(kernel_matrix_hk)
# Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
cholesky_kernel_matrix_hk = np.linalg.cholesky(kernel_matrix_hk + 1e-6 * np.eye(len(X) * len(X)))

# Sample 3 standard normals for each of the unseen data points
standard_normals = np.random.normal(size=(len(X) * len(X), 2))

# multiply them by the square root of the covariance matrix L
kernel_samples = np.dot(cholesky_kernel_matrix_hk, standard_normals)
exit(0)

###########################################

n = 10
standard_normals = np.random.normal(size=(n, 1))
print(standard_normals.shape)

exit(0)

cola = 2
rowa = 2
colb = 2
rowb = 2


def my_kron(A, B):
    C = [[0 for j in range(cola * colb)] for i in range(rowa * rowb)]
    for i in range(0, rowa * rowb):
        for j in range(0, cola * colb):
            row_index1 = int(i / rowb)
            col_index1 = int(j / colb)
            row_index2 = i % rowb
            col_index2 = j % colb
            C[i][j] = A[row_index1][col_index1] * B[row_index2][col_index2]
        print("\n")
    print(C)


def Kroneckerproduct(A, B):
    C = [[0 for j in range(cola * colb)] for i in range(rowa * rowb)]

    # i loops till rowa
    for i in range(0, rowa):

        # k loops till rowb
        for k in range(0, rowb):

            # j loops till cola
            for j in range(0, cola):

                # l loops till colb
                for l in range(0, colb):
                    # Each element of matrix A is
                    # multiplied by whole Matrix B
                    # resp and stored as Matrix C
                    # C[i + l +1][j + k +1] = A[i][j] * B[k][l]
                    # print("i=",i + l + 1,"j=",j + k + 1, C[i + l + 1][j + k + 1], end=' \t')
                    print(A[i][j] * B[k][l], "-->", i + l + 1, j + k + 1)
            print("\n")
    print("\n\n\n\nhere", C)

    # Driver code.


# rowa and cola are no of rows and columns
# of matrix A
# rowb and colb are no of rows and columns
# of matrix B


# # Function to computes the Kronecker Product
# # of two matrices
#
#
#
# A = [[0 for j in range(2)] for i in range(3)]
# B = [[0 for j in range(3)] for i in range(2)]
#
# A[0][0] = 1
# A[0][1] = 2
# A[1][0] = 3
# A[1][1] = 4
# A[2][0] = 1
# A[2][1] = 0
#
# B[0][0] = 0
# B[0][1] = 5
# B[0][2] = 2
# B[1][0] = 6
# B[1][1] = 7
# B[1][2] = 3

A = np.linspace(1, 4, 4).reshape(2, 2)
B = np.linspace(1, 4, 4).reshape(2, 2)
print(B)

# Kroneckerproduct(A, B)
my_kron(A, B)
