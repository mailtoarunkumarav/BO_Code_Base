import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import datetime
import sys

# kernel_type = 0 #
# number_of_test_datapoints = 20
# np.random.seed(500)
# noise = 0.0


# kernel_type = 0
# number_of_test_datapoints = 20
np.random.seed(400)
# noise = 0.0

class Custom_Print(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self) :
        for f in self.files:
            f.flush()



# Class to handle the Gaussian Process related tasks required for the Bayesian Optimization
class GaussianProcess:

    # Initializing the Gaussian Process object with the predefined GP Settings as specified by the user
    def __init__(self, kernel_type, params_estimation, len_scale_estimation, char_length_scale, len_scale_bounds,
                 signal_variance, signal_variance_bounds,
                 number_of_test_datapoints, noise, random_seed, linspacexmin, linspacexmax,
                 linspaceymin, linspaceymax, bounds, number_of_dimensions, number_of_observed_samples, kernel_char,
                 len_scale_params, len_scale_param_bounds, len_scale_func_type, number_of_restarts_likelihood, Xmin, Xmax, ymin, ymax,
                 len_weights_bounds, len_weights, weights_estimation, multi_len_scales, beta_bounds, beta_params):

        self.kernel_type = kernel_type
        self.params_estimation = params_estimation
        self.char_length_scale = char_length_scale
        self.len_scale_bounds = len_scale_bounds
        self.number_of_test_datapoints = number_of_test_datapoints
        self.noise = noise
        self.linspacexmin = linspacexmin
        self.linspacexmax = linspacexmax
        self.linspaceymin = linspaceymin
        self.linspaceymax = linspaceymax
        # not required as we are generating random samples at each run
        # np.random.seed(random_seed)
        self.bounds = bounds
        self.number_of_dimensions = number_of_dimensions
        self.number_of_observed_samples = number_of_observed_samples
        self.signal_variance = signal_variance
        self.signal_variance_bounds = signal_variance_bounds
        self.L_x_x = np.zeros(number_of_dimensions)
        self.kernel_char = kernel_char
        self.len_scale_params = len_scale_params
        self.len_scale_param_bounds = len_scale_param_bounds
        self.len_scale_estimation = len_scale_estimation
        self.number_of_restarts_likelihood = number_of_restarts_likelihood
        self.len_scale_func_type = len_scale_func_type
        self.Xmax = Xmax
        self.Xmin = Xmin
        self.ymin = ymin
        self.ymax = ymax
        self.len_weights_bounds = len_weights_bounds
        self.len_weights = len_weights
        self.weights_estimation = weights_estimation
        self.multi_len_scales = multi_len_scales
        self.disp_bool = True

        #Regression Addition
        self.beta_bounds = beta_bounds
        self.beta_params = beta_params


    # Method to set the model used by the Gaussian Process
    def gaussian_fit(self, X, y):

        # Update the contents of X and y
        self.X = X
        # y_mean = y.mean()
        # y_std = np.sqrt(y.var())
        # y = y-y_mean/y_std
        self.y = y
        # Recalculating L with updated length scale
        # Commented to check the integrity of the modified code in the regression setting
        self.L_x_x = self.compute_l(X)
        print("L Recaluated with new data")

    # Define the kernel function to be used in the GP
    def computekernel(self, data_point1, data_point2):

        # Depending on the setting specified by the user, appropriate kernel function is used in the Gaussian Process

        # Kernel_type = 0 represents the Squared Exponential Kernel
        if self.kernel_type == 0:
            # print("SE Kernel")
            # result = self.sq_exp_kernel(data_point1, data_point2, self.char_length_scale, self.signal_variance)
            result = self.free_kernel(data_point1, data_point2, self.char_length_scale, self.signal_variance)

        # Kernel_type = 1 represents the Rational Quadratic Function Kernel
        elif self.kernel_type == 1:
            print("RQF Kernel")
            # self.charac_length_scale = 1
            alpha = 0.1
            result = self.rational_quadratic_kernel(data_point1, data_point2, self.char_length_scale, alpha)

        # Kernel_type = 2 represents the Exponential Kernel
        elif self.kernel_type == 2:
            print("EXP Kernel")
            # self.charac_length_scale = 0.1
            result = self.exp_kernel_function(data_point1, data_point2, self.char_length_scale)

        # Kernel_type = 3 represents the Periodic Kernel***(To be analysed for fixin the parameters)
        elif self.kernel_type == 3:
            print("Periodic Kernel")
            # self.charac_length_scale = 0.1
            result = self.periodic_kernel_function(data_point1, data_point2, self.char_length_scale)

        return result

    def sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        if (self.kernel_char == 'ard' or self.kernel_char == 'fix_l'):

            # if(self.params_estimation == True):
            #     print("ARD kernel is set for the computations")
            # else:
            # print("Fixed length kernel is set for the computations")
            return self.ard_sq_exp_kernel(data_point1, data_point2, char_len_scale, signal_variance)

        elif (self.kernel_char == 'var_l'):
            # print("Var kernel is set for the computations")
            return self.var_sq_exp_kernel(data_point1, data_point2, char_len_scale, signal_variance)

        elif (self.kernel_char == 'm_ker'):
            # print("Var kernel is set for the computations")
            return self.multi_sq_exp_kernel(data_point1, data_point2, self.multi_len_scales, signal_variance)

    def ard_sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        # Implements Automatic Relevance Determinations (ARD) Kernel

        # k(x1,x2) = sig_squared * exp{(-1/2*(datapoint1 - datapoint2) * M2 * (datapoint1 - datapoint2).T))}
        # M1  =  l^(-2)*I, M2 = diag(l)^(-2) , M3 = ones * ones.T + diag(l)^(-2)

        # Element wise squaring the vector of given length scales
        char_len_scale = np.array(char_len_scale) ** 2

        # Creating a Diagonal matrix with squared l values
        sq_dia_len = np.diag(char_len_scale)

        # Computing inverse of a diagonal matrix by reciprocating each item in the diagonal
        inv_sq_dia_len = np.linalg.pinv(sq_dia_len)
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = ((data_point1[i, :] - data_point2[j, :]))
                product1 = np.dot(difference, inv_sq_dia_len)
                final_product = np.dot(product1, difference.T)
                each_kernel_val = (signal_variance ** 2) * (np.exp((-1 / 2.0) * final_product))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def len_scale_func_linear(self, data_point_value, len_scale):

        a = len_scale[0]
        b = len_scale[1]

        value = a * data_point_value + b

        if value == 0:
            value = 1e-6

        return value

    def len_scale_func_gauss(self, data_point_value, len_scale):

        bias = 0
        mean = len_scale[0]
        std_dev = len_scale[1]

        exp_term = np.exp((-0.5) * (((data_point_value - mean) / std_dev) ** 2))
        pre_term = 1 / np.sqrt(2 * np.pi * (std_dev ** 2))
        # pre_term = 1
        value = pre_term * exp_term + bias

        if value == 0:
            value = 1e-6

        return value

    def len_scale_func_quad(self, data_point_value, len_scale):

        a = len_scale[0]
        b = len_scale[1]
        c = len_scale[2]

        # #commented for convex parabola
        value = a * (data_point_value ** 2) + b * data_point_value + c
        if value == 0:
            value = 1e-6
        if value < 0:
            print("quad length scale value is less than zero")
        return value

    def var_sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        # Implements the spatially varying length scale

        # Commenting the following block as it is not required if spatially varying length scale is not computed
        # Creating a Diagonal matrix with squared l values
        # sq_dia_len = np.diag(char_len_scale)
        # Computing inverse of a diagonal matrix by reciprocating each item in the diagonal
        # inv_sq_dia_len = np.linalg.pinv(sq_dia_len)

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))

        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                len_scale_vectors = []
                for d in np.arange(self.number_of_dimensions):

                    if self.len_scale_func_type[d] == 'linear':
                        len_scale_vector_datapoint1 = self.len_scale_func_linear(data_point1[i][d], self.len_scale_params[d])
                        len_scale_vector_datapoint2 = self.len_scale_func_linear(data_point2[j][d], self.len_scale_params[d])

                    elif self.len_scale_func_type[d] == 'gaussian':
                        len_scale_vector_datapoint1 = self.len_scale_func_gauss(data_point1[i][d], self.len_scale_params[d])
                        len_scale_vector_datapoint2 = self.len_scale_func_gauss(data_point2[j][d], self.len_scale_params[d])

                    elif self.len_scale_func_type[d] == 'quadratic':
                        len_scale_vector_datapoint1 = self.len_scale_func_quad(data_point1[i][d], self.len_scale_params[d])
                        len_scale_vector_datapoint2 = self.len_scale_func_quad(data_point2[j][d], self.len_scale_params[d])

                    len_scale_vectors.append([len_scale_vector_datapoint1, len_scale_vector_datapoint2])

                difference = data_point1[i, :] - data_point2[j, :]
                total_product = 1
                total_sum = 0

                for k in np.arange(self.number_of_dimensions):
                    denominator = len_scale_vectors[k][0] ** 2 + len_scale_vectors[k][1] ** 2
                    total_product *= (2 * len_scale_vectors[k][0] * len_scale_vectors[k][1]) / denominator
                    total_sum += 1 / denominator
                if (total_product < 0):
                    print("Product term of length scale is less than zero", data_point1, data_point2)

                squared_diff = np.dot(difference, difference.T)
                each_kernel_val = (signal_variance ** 2) * np.sqrt(total_product) * (np.exp((-1) * squared_diff * total_sum))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    ## Implementing multi Kernel
    def multi_sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = ((data_point1[i, :] - data_point2[j, :]))
                sq_difference = np.dot(difference, difference.T)
                each_kernel_val = 0
                for count in np.arange(4):
                    each_kernel_val += self.len_weights[count] * (signal_variance ** 2) * \
                                       (np.exp(-0.5 * sq_difference * (1 / (char_len_scale[count] ** 2))))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def len_scale_func(self, data_point):

        # # Initializations for a and b
        # Linear Functions a.x + b
        # a = 0.085; b = 0.1;         # l : x + 1
        # a = -0.080; b = 0.9;         # l : x + 1
        # a = 1; b = 2;         # l : x + 2
        # a = 2; b = 1;         # l : 2x + 1

        # Quadratic Functions a.x^2 + b.x+c
        # a = 2; b = 1; c = 0         # l : x^2 + x
        # a = 1; b = 1; c = 0         # l : x^2 + x
        # a = 1; b = 0; c = 0         # l : x^2
        # a = -1/30; b = 1/3; c = 1/15 # Concave downward or concave or convex upward
        # a = 1 / 30; b = -1 / 3; c = 23 / 24  # Convex downward or convex or concave upward

        a = self.len_scale_params[0]
        b = self.len_scale_params[1]
        c = 0

        # len_scale_weights = np.zeros(data_point.shape)
        len_scale_weights = np.array([])
        len_scale_values = np.array([])
        data_point_values = np.array([])

        for dim_count in np.arange(len(data_point)):
            # in case if it varies quadratically in each dimension
            # if(dim_count == 1 ):
            #     len_scale_values[dim_count] = a * data_point[dim_count] + b * (data_point[dim_count] ** 2)
            # if(dim_count == 2 ):
            #     len_scale_values[dim_count] = a * data_point[dim_count] + b * data_point[dim_count]
            # linearly varying length scale with respect to dimensions
            # len_scale_weights[dim_count] = a

            # # # Linear calculations
            # len_scale_weights = np.append(len_scale_weights, a)
            # data_point_values = np.append(data_point_values, data_point[0])
            # value = np.dot(len_scale_weights.T, data_point) + b

            # Quadratic calculations
            len_scale_weights = np.append(len_scale_weights, a)
            len_scale_weights = np.append(len_scale_weights, b)
            data_point_values = np.append(data_point_values, data_point[0] ** 2)
            data_point_values = np.append(data_point_values, data_point[0])
            value = np.dot(len_scale_weights.T, data_point_values) + c

            if value == 0:
                value = 1e-6
            len_scale_values = np.append(len_scale_values, value)

        # Logistic function for length scale functions
        # len_scale_values = np.array([])
        # value = (1/(1.25+np.exp(5-data_point)))+ 0.1
        # len_scale_values = np.append(len_scale_values, value)

        # Gaussian function for the the length scale
        # concave_min = True
        # # concave_min = False
        # len_scale_values = np.array([])
        #
        # if concave_min:
        #     std_dev = 0.3;
        #     bias = 0.05;
        #     mu = 1;
        #     concave_convex = 1
        #     pre_term = 1 / np.sqrt(2 * np.pi * (std_dev ** 2))
        #     exp_term = np.exp((-0.5) * (((data_point - mu) / std_dev) ** 2))
        # else:
        #     std_dev = 0.4;
        #     bias = 1.05;
        #     mu = 1;
        #     concave_convex = -1
        #     pre_term = 1 / np.sqrt(2 * np.pi * (std_dev ** 2))
        #     exp_term = np.exp((-0.5) * (((data_point - mu) / std_dev) ** 2))
        #
        # value = concave_convex * pre_term * exp_term + bias
        # len_scale_values = np.append(len_scale_values, value)

        return len_scale_values

    # Other Kernel Functions
    def rational_quadratic_kernel(self, data_point1, data_point2, charac_length_scale, alpha):

        # Define Rational Quadratic Function
        # k(x1,x2) = (1 + ((total_squared_distances(x1,x2) / (2.0 * (charac_length_scale ** 2) * alpha)))) ** (-alpha)
        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = (1 + ((total_squared_distances / (2.0 * (charac_length_scale ** 2) * alpha)))) ** (-alpha)
        return kernel_val

    def exp_kernel_function(self, data_point1, data_point2, charac_length_scale):

        # exponential covariance function , special case of matern with v = 1/2
        # k(x1,x2) = exp{-(abs(x2 - x1) / charac_length_scale)}
        kernel_val = np.exp(-(abs(data_point2 - data_point1) / charac_length_scale))
        return kernel_val

    def periodic_kernel_function(self, data_point1, data_point2, charac_length_scale):

        # Periodic covariance function
        # k(x1,x2) = exp{-2.0 * (sin(pi * (x2 - x1))) ** 2 * (1 / charac_length_scale ** 2)}
        kernel_val = np.exp(-2.0 * (np.sin(np.pi * (data_point2 - data_point1))) ** 2 * (1 / charac_length_scale ** 2))
        return kernel_val
    # Estimating kernel parameters
    def optimize_log_marginal_likelihood_l(self, input):
        # 0 to n-1 elements represent the nth eleme
        init_charac_length_scale = np.array(input[: self.number_of_dimensions])
        signal_variance = input[len(input) - 1]
        K_x_x = self.sq_exp_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        eye = 1e-6 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood

    # Estimating kernel parameters
    def optimize_log_marginal_likelihood_l_params(self, input):
        # 0 to n-1 elements represent the nth eleme

        total_len_scale_params = []
        param_count = 0
        for type in self.len_scale_func_type:
            if (type == 'linear' or type == 'gaussian'):
                total_len_scale_params.append(input[param_count:(param_count + 2)])
                param_count += 2
            elif (type == 'quadratic'):
                total_len_scale_params.append(input[param_count:(param_count + 3)])
                param_count += 3

        self.len_scale_params = total_len_scale_params

        # Following parameters not used in any computations
        init_charac_length_scale = 0.1
        signal_variance = input[len(input) - 1]
        self.signal_variance = signal_variance

        K_x_x = self.sq_exp_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        eye = 1e-10 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        try:
            L_x_x = np.linalg.cholesky(Knoise)
            factor = np.linalg.solve(L_x_x, self.y)
            products = np.dot(factor.T, factor)

        except np.linalg.LinAlgError:
            # print("!!!!!!!!!!!Matrix is not positive definite, inverting with pinv\nEigen", np.linalg.eigvals(Knoise))
            # print("len", self.len_scale_params, self.signal_variance,self.X,self.y )
            if self.disp_bool:
                print("!!!!!!!!!!!Matrix is not positive definite, inverting with pinv\nEigen", np.linalg.eigvals(Knoise), "\n", K_x_x)
            self.disp_bool = False
            K_pinv = np.linalg.pinv(Knoise)
            factor_pinv = np.dot(self.y.T, K_pinv)
            products = np.dot(factor_pinv, self.y)

        log_marginal_likelihood = -0.5 * (products + self.number_of_observed_samples * np.log(2 * np.pi) + np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood

    # Estimating kernel parameters
    def optimize_log_marginal_likelihood_weight_params(self, input):

        self.len_weights = input[0:4]
        # Following parameters not used in any computations
        multi_len_scales = self.multi_len_scales
        signal_variance = input[len(input) - 1]
        self.signal_variance = signal_variance

        K_x_x = self.sq_exp_kernel(self.X, self.X, multi_len_scales, signal_variance)
        eye = 1e-6 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        try:
            L_x_x = np.linalg.cholesky(Knoise)

        except np.linalg.LinAlgError:
            print("Matrix is not positive definite here")

        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood


    #Regression addition
    def free_kernel(self, data_point1, data_point2, charac_length_scale, signal_variance):

        numerator = self.free_kernel_wrapper(data_point1, data_point2, charac_length_scale, signal_variance)
        denom1 = self.free_kernel_wrapper(data_point1, data_point1, charac_length_scale, signal_variance)
        denom2 = self.free_kernel_wrapper(data_point2, data_point2, charac_length_scale, signal_variance)

        # if(np.any (denom1) or np.any(denom2) ):
        #     print(denom1)
        #     print("zeroes encountered ... ")

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                kernel_mat[i,j] = numerator[i,j]/(np.sqrt(denom1[i,i] * denom2[j,j]))

        return kernel_mat

    def free_kernel_wrapper(self, data_point1, data_point2, charac_length_scale, signal_variance):
        random_vector1 = [0.2]
        random_vector2 = [0.5]

        operand1 = (self.beta_params[0] ** 2) * self.free_se_kernel(data_point1, data_point2, random_vector1, random_vector1,
                                                                    charac_length_scale, signal_variance)
        operand2 = ((self.beta_params[0]) * (self.beta_params[1])) * self.free_se_kernel(data_point1, data_point2,
                                                                                                    random_vector1,
                                                                                                    random_vector2,
                                                                    charac_length_scale, signal_variance)
        operand3 = ((self.beta_params[1]) * (self.beta_params[0])) * self.free_se_kernel(data_point1, data_point2,
                                                                                                    random_vector2,
                                                                                                    random_vector1,
                                                                    charac_length_scale, signal_variance)
        operand4 = ((self.beta_params[1]) ** 2) * self.free_se_kernel(data_point1, data_point2, random_vector2, random_vector2,
                                                                    charac_length_scale, signal_variance)
        kernel_val = operand1 + operand2 + operand3 + operand4

        return kernel_val

    def free_se_kernel(self, data_point1, data_point2, rand_vector1, rand_vector2, charac_length_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                each_kernel_val = np.multiply(data_point1[i, :], data_point2[j, :])
                rand_vector_product = np.multiply(rand_vector1, rand_vector2)
                total_product = np.multiply(each_kernel_val, rand_vector_product)
                each_kernel_val = np.sum(total_product)
                term2 = ( 2 * each_kernel_val - ((data_point1[i, :]) **2)- ((
                              data_point2[j,:]) **2) - (np.square(rand_vector1))- (np.square(rand_vector2)))
                kernel_mat[i, j] = np.exp(0.5 * (1/(charac_length_scale**2)) * term2) * signal_variance

        return kernel_mat

    def optimize_log_marginal_likelihood_l(self, input):
        # 0 to n-1 elements represent the nth eleme

        self.beta_params = np.array(input[:2])
        self.signal_variance = input[len(input) - 1]
        # self.charac_length_scale = input[2: 2+self.number_of_dimensions][0]
        # K_x_x = self.sq_exp_kernel(self.X, self.X, self.charac_length_scale, signal_variance)
        init_charac_length_scale = input[2: 2+self.number_of_dimensions]
        K_x_x = self.free_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        eye = 1e-12 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood


    def compute_l(self, X):
        # Apply the kernel function to find covariances between our observed points
        K_x_x = self.computekernel(X, X)
        # print("x: ", X, "\nK: ",K_x_x )
        # Add some noise to avoid decay of eigen vectors to avoid going into non positive definite matrix
        eye = 1e-6 * np.eye(len(X))
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(K_x_x + eye)
        self.L_x_x = L_x_x
        return L_x_x

    # Compute mean and variance required for the calculation of posteriors
    def compute_mean_var(self, Xs, X, y):

        # Compute Mean for the training data points
        # mu = (K_x_xs.T) (inv(K_x_x)) (y)
        # mu = (K_x_xs.T) inv(((L)(L.T))) (y)  #since K = (L)(L.T)
        # mu = (K_x_xs.T) (inv(L.T))(inv(L)) (y)
        # mu = ((K_x_xs.T) (inv(L).T)) ((inv(L)) (y)) # since inv(A.T) = (inv(A)).T
        # mu = (((inv(L) (K_x_xs)).T)) ((inv(L)) (y))
        # mu = ((L\K_x_xs).T)(L\y)  # since (inv)A (b) = A\b of Ax =b

        # Apply the kernel function to find covariances between the unseen data points and the observed samples
        K_x_xs = self.computekernel(X, Xs)
        factor1 = np.linalg.solve(self.L_x_x, K_x_xs)
        factor2 = np.linalg.solve(self.L_x_x, y)
        #commented for regression setting
        # mean = np.dot(factor1.T, factor2)
        mean = np.dot(factor1.T, factor2).flatten()

        # compute variance at our test points
        # var = K_xs_xs - (K_x_xs.T) (inv(K_x_x)) (K_x_xs)
        # var = K_xs_xs - (K_x_xs.T) (inv((L)(L.T))) (K_x_xs)
        # var = K_xs_xs - (K_x_xs.T) (inv(L.T))(inv(L)) (K_x_xs)
        # var = K_xs_xs - ((K_x_xs.T) (inv(L.T)))((inv(L)) (K_x_xs))
        # var = K_xs_xs - ((K_x_xs.T) (inv(L.T)))((inv(L)) (K_x_xs))
        # var = K_xs_xs - (((inv(L)) (K_x_xs)).T)((inv(L)) (K_x_xs))
        # var = K_xs_xs - (V.T)(V) # since V = ((inv(L)) (K_x_xs))

        # Applying kernel function to find covariances between the unseen datapoints to find variance
        K_xs_xs = self.computekernel(Xs, Xs)
        variance = K_xs_xs - np.dot(factor1.T, factor1)

        return mean, variance, factor1

    # Method used to predict the mean and variance for the unseen data points
    def gaussian_predict(self, Xs):

        # compute the covariances between the unseen data points i.e K**
        K_xs_xs = self.computekernel(Xs, Xs)

        # Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
        L_xs_xs = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints))

        # Sample 3 standard normals for each of the unseen data points
        standard_normals = np.random.normal(size=(self.number_of_test_datapoints, 3))

        # multiply them by the square root of the covariance matrix L
        f_prior = np.dot(L_xs_xs, standard_normals)

        # Compute mean and variance
        mean, variance, factor1 = self.compute_mean_var(Xs, self.X, self.y)
        diag_variance = np.diag(variance)
        # standard_deviation = np.sqrt(diag_variance)

        # compute posteriors for the data points
        # newL = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints) - np.dot(factor1.T, factor1))
        # f_post = mean.reshape(-1, 1) + np.dot(newL, np.random.normal(size=(self.number_of_test_datapoints, 3)))
        f_post = None
        return mean, diag_variance, f_prior, f_post


    def plot_graph(self, plot_params):
        plt.figure(plot_params['plotnum'])
        plt.clf()
        for eachplot in plot_params['plotvalues']:
            if (len(eachplot) == 2):
                plt.plot(eachplot[0], eachplot[1])
            elif (len(eachplot) == 3):
                plt.plot(eachplot[0], eachplot[1], eachplot[2])
            elif (len(eachplot) == 4):
                if(eachplot[3].startswith("label=")):
                    plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:])
                    plt.legend(loc='upper right',prop={'size': 6})
                else:
                    flag = eachplot[3]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], lw=eachplot[3][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], ms=eachplot[3][2:])

            elif (len(eachplot) == 5):

                if(eachplot[3].startswith("label=")):
                    flag = eachplot[4]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:], lw=eachplot[4][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:], ms=eachplot[4][2:])
                    plt.legend(loc='upper right',prop={'size': 6})

                else:
                    flag = eachplot[3]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], lw=eachplot[3][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], ms=eachplot[3][2:])

        if 'gca_fill' in plot_params.keys():
            if len(plot_params['gca_fill']) == 3:
                plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                       plot_params['gca_fill'][2],
                                       color="#ddddee")
            else:
                if plot_params['gca_fill'][3].startswith('color'):
                    color = plot_params['gca_fill'][3][6:]
                    print(len(plot_params['gca_fill']), color)
                    plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                           plot_params['gca_fill'][2], color=color)

        plt.axis(plot_params['axis'])
        plt.title(plot_params['title'])
        plt.xlabel(plot_params['xlabel'])
        plt.ylabel(plot_params['ylabel'])
        plt.savefig(plot_params['file'], bbox_inches='tight')


    def plot_graph_old(self, plot_params):

        ''' Generic method to plot 1D graphs according to the values parameters passed in the parameter
           Plot params can be specified as shown in the example below
           plot_params = {'plotnum': 'Fig 3-'+str(count),
                                              'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,
                                                       self.linspaceymax],
                                              'plotvalues': [[self.X, self.y, 'r+', 'ms15'], [Xs, ys, 'b-'], [Xs, mean,
                                                                                                              'g--', 'lw2']],
                                              'title': 'GP Posterior Distribution with length scale L= ' + str(
                                                  self.char_length_scale),
                                              'file': 'GP_Posterior_Distr'+str(count),
                                              'gca_fill': [Xs.flat, (mean.flatten() - 2 * standard_deviation).reshape(-1,1).flat,
                                                           (mean.flatten() + 2 * standard_deviation).reshape(-1,1).flat]
                                              }
        '''

        # Sets the name of the figure
        plt.figure(plot_params['plotnum'])

        # Clear the graph if any junk data is present from previous usages
        plt.clf()

        # For each of the plotting parameters specified, construct the plot accordingly
        for eachplot in plot_params['plotvalues']:

            # Plotting when only X axis and Y Axis values are specified as parameters
            if (len(eachplot) == 2):
                plt.plot(eachplot[0], eachplot[1])

            # Used when extra parameters like linewidth(lw) or marker size (ms) is specified as parameter
            elif (len(eachplot) == 3):
                plt.plot(eachplot[0], eachplot[1], eachplot[2])

            # Multiple parameters passed for plotting
            elif (len(eachplot) == 4):
                flag = eachplot[3]
                if flag.startswith('lw'):
                    plt.plot(eachplot[0], eachplot[1], eachplot[2], lw=eachplot[3][2:])
                elif flag.startswith('ms'):
                    plt.plot(eachplot[0], eachplot[1], eachplot[2], ms=eachplot[3][2:])

        # Executed when there is a required to fill some region in order to indicate the deviations or errors
        if 'gca_fill' in plot_params.keys():

            # Depending on the parameters of the filling passed, appropriate block is called to render the graph
            if len(plot_params['gca_fill']) == 3:
                plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                       plot_params['gca_fill'][2],
                                       color="#ddddee")
            else:
                if plot_params['gca_fill'][3].startswith('color'):
                    color = plot_params['gca_fill'][3][6:]
                    print(len(plot_params['gca_fill']), color)
                    plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                           plot_params['gca_fill'][2], color=color)

        # Set the parameters of the graph being plotted
        plt.axis(plot_params['axis'])
        plt.title(plot_params['title'])
        plt.savefig(plot_params['file'], bbox_inches='tight')

    # Method used to plot the Gaussian prior with the specified f_prior in the case of 1D problem
    def plot_prior_samples(self, Xs, f_prior):

        # Specify the parameters required for plotting the prior
        plot_prior_params = {'plotnum': 'Fig 1',
                             'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin, self.linspaceymax],
                             'plotvalues': [[Xs, f_prior], [Xs, np.zeros(len(Xs)), 'b--', 'lw2']],
                             'title': 'GP Prior Samples',
                             'file': 'GP_Prior'
                             }
        self.plot_graph(plot_prior_params)

    # Method used to plot posteriors with the specified f_post in the case of 1D problem
    def plot_posterior_samples(self, Xs, f_post):

        plot_posterior_sample_params = {'plotnum': 'Fig 2',
                                        'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,
                                                 self.linspaceymax],
                                        'plotvalues': [[self.X, self.y, 'r+', 'ms15'], [Xs, f_post]],
                                        'title': 'GP Posterior Samples',
                                        'file': 'GP_Posterior_Samples'
                                        }

        self.plot_graph(plot_posterior_sample_params)

    # Method used to plot the predictions in the case of 1D problem with the mean and standard deviations
    # and function's evaluations at observed samples as well as predictions from Gaussian Process
    def plot_posterior_predictions(self, count, Xs, ys, mean, standard_deviation):

        plot_posterior_distr_params = {'plotnum': 'Fig 3-' + str(count),
                                       # 'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,self.linspaceymax],
                                       ## Normalizing code
                                       'axis': [0, 1, 0, 1],
                                       'plotvalues': [[self.X, self.y, 'r+', 'ms15'], [Xs, ys, 'b-'], [Xs, mean,
                                                                                                       'g--', 'lw2']],
                                       'title': 'GP Posterior Distr. with Spatially varying length scale'
                                       # + str(self.char_length_scale)
            , 'file': 'GP_Posterior_Distr' + str(count),
                                       'gca_fill': [Xs.flat, (mean.flatten() - 2 * standard_deviation).reshape(-1, 1).flat,
                                                    (mean.flatten() + 2 * standard_deviation).reshape(-1, 1).flat]
                                       }

        self.plot_graph(plot_posterior_distr_params)

    # Helper method to plot prior, posterior samples and predictions in the case of 1D problem
    def plot_visuals(self, run_count, Xs, ys, mean, standard_deviation, f_prior, f_post):

        # self.plot_prior_samples(Xs, f_prior)
        # self.plot_posterior_samples(Xs, f_post)
        self.plot_posterior_predictions(run_count, Xs, ys, mean, standard_deviation)

    # Method to predict the values for the unknown function at unseen data points
    # and plot prior, posterior and predictions simultaneously
    def gaussian_predict_plot(self, Xs, ys):

        # compute the covariances between the test data points i.e K**
        K_xs_xs = self.computekernel(Xs, Xs)

        # Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
        L_xs_xs = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints))

        # Sample 3 standard normals for each of the unseen data points
        standard_normals = np.random.normal(size=(self.number_of_test_datapoints, 3))

        # multiply them by the square root of the covariance matrix L
        f_prior = np.dot(L_xs_xs, standard_normals)

        # Set parameters to plot gaussian priors
        plot_prior_params = {'plotnum': 'Fig 1_',
                             'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin, self.linspaceymax],
                             'plotvalues': [[Xs, f_prior], [Xs, np.zeros(len(Xs)), 'b--', 'lw2']],
                             'title': 'GP Prior Samples',
                             'file': 'GP_Prior'
                             }
        self.plot_graph(plot_prior_params)

        # Compute mean, variance to calculate posterior distributions
        mean, variance, factor1 = self.compute_mean_var(Xs, self.X, self.y)
        diag_variance = np.diag(variance)
        standard_deviation = np.sqrt(diag_variance)

        # compute posterior for the data points
        newL = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints) - np.dot(factor1.T, factor1))
        f_post = mean.reshape(-1, 1) + np.dot(newL, np.random.normal(size=(self.number_of_test_datapoints, 3)))

        # Setting parameters to plot posterior samples
        plot_posterior_sample_params = {'plotnum': 'Fig 2_',
                                        'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,
                                                 self.linspaceymax],
                                        'plotvalues': [[self.X, self.y, 'r+', 'ms20'], [Xs, f_post]],
                                        'title': 'GP Posterior Samples',
                                        'file': 'GP_Posterior_Samples'
                                        }
        self.plot_graph(plot_posterior_sample_params)

        # Setting parameters to plot posterior distributions
        plot_posterior_distr_params = {'plotnum': 'Fig 3_',
                                       'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,
                                                self.linspaceymax],
                                       'plotvalues': [[self.X, self.y, 'r+', 'ms20'], [Xs, ys, 'b-'], [Xs, mean,
                                                                                                       'r--', 'lw2']],
                                       'title': 'GP Posterior Distribution with length scale L= ' + str(
                                           self.char_length_scale),
                                       'file': 'GP_Posterior_Distr',
                                       'gca_fill': [Xs.flat, mean - 2 * standard_deviation,
                                                    mean + 2 * standard_deviation]
                                       }
        self.plot_graph(plot_posterior_distr_params)

        return mean, variance

    # Archived versions of squared exponential kernel
    def sq_exp_kernel_arxiv(self, data_point1, data_point2, charac_length_scale, signal_variance):

        # k(x1,x2) = sig_squared*exp{(-1/2*(charac_length_scale))((euclidean_distance(x1,x2)**2))}
        # Parameter sig_squared = 1
        # distances((a1,a2),(b1,b2)) = sqrt(a1**2 + a2**2 + b1**2 + b2**2 + 2(a1*b1 + a2*b2))
        total_squared_distances = np.sum(np.square(data_point1), 1).reshape(-1, 1) + np.sum(np.square(data_point2), 1) \
                                  - 2 * np.dot(data_point1, data_point2.T)
        kernel_val = signal_variance * (np.exp(-(total_squared_distances * (1 / ((charac_length_scale ** 2) * 2.0)))))
        return kernel_val


    def runGaussian(self, count, Xs, ys, X, y):

        print("!!!!!!!!!!Gaussian Process: Iteration", count, " Started!!!!!!!!!")

        # if (self.likelihood_estimation):
        if (True):

            # Estimating Length scale itself
            x_max_value = None
            log_like_max =  -1* float("inf")

            # Data structure to create the starting points for the scipy.minimize method
            random_points_lengthscale = []

            beta1_start_points = np.random.uniform(self.beta_bounds[0][0],
                                                   self.beta_bounds[0][1],
                                                      self.number_of_restarts_likelihood)

            beta2_start_points = np.random.uniform(self.beta_bounds[1][0],
                                                   self.beta_bounds[1][1],
                                                   self.number_of_restarts_likelihood)

            ##Commenting the code here as ARD is not required for the calculations
            # for dim in np.arange(self.number_of_dimensions):
            #     random_points_lengthscale_eachdim = np.random.uniform(self.lenscale_bounds[dim][0],
            #                                               self.lenscale_bounds[dim][1],
            #                                               self.number_of_restarts_likelihood). \
            #         reshape(1, self.number_of_restarts_likelihood)
            #     random_points_lengthscale.append(random_points_lengthscale_eachdim)
            #
            # # Vertically stack the arrays of randomly generated starting points as a matrix
            #     random_points_lengthscale = np.vstack(random_points_lengthscale)

            random_points_lengthscale_eachdim = np.random.uniform(self.len_scale_bounds[0][0],
                                                                  self.len_scale_bounds[0][1],
                                                                  self.number_of_restarts_likelihood). \
                reshape(1, self.number_of_restarts_likelihood)
            random_points_lengthscale.append(random_points_lengthscale_eachdim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_lengthscale = np.vstack(random_points_lengthscale)

            variance_start_points = np.random.uniform(self.signal_variance_bounds[0],
                                                      self.signal_variance_bounds[1],
                                                      self.number_of_restarts_likelihood)

            total_bounds = self.beta_bounds.copy()
            total_bounds.append(self.len_scale_bounds[0])
            total_bounds.append(self.signal_variance_bounds)

            for ind in np.arange(self.number_of_restarts_likelihood):

                if(ind == 4):
                    print(" debugging .. ")

                beta1 = beta1_start_points[ind]
                beta2 = beta2_start_points[ind]
                l_init = random_points_lengthscale[0][ind]
                var_init = variance_start_points[ind]
                init_points = []
                init_points.append(beta1)
                init_points.append(beta2)
                init_points.append(l_init)
                init_points.append(var_init)
                print(ind+1, ". Initial values for beta1 & beta2 are ", beta1, beta2, l_init, var_init)
                maxima =  opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_l(x),
                                      init_points,
                                      method='L-BFGS-B',
                                      bounds=total_bounds
                                      ,options={ 'maxfun': 2000, 'maxiter': 2000}
                                       )

                if not maxima['success']:
                    print ("Error occured : ", maxima['message'], "Log L: ", -1*maxima['fun'])
                    continue

                beta1_temp = maxima['x'][0]
                beta2_temp = maxima['x'][1]
                l_temp = maxima['x'][2]
                var_temp = maxima['x'][3]
                params = np.append(beta1_temp, beta2_temp)

                log_likelihood = -1* maxima['fun']

                print("Comparing", log_likelihood, log_like_max)
                if (log_likelihood > log_like_max):
                    print("New maximum log likelihood ", -1 * log_likelihood, " found for params ", params, l_temp,var_temp)
                    x_max_value = maxima
                    log_like_max = log_likelihood

            beta1 = x_max_value['x'][0]
            beta2 = x_max_value['x'][1]
            self.char_length_scale = x_max_value['x'][2]
            self.signal_variance = x_max_value['x'][3]
            self.beta_params = np.array([beta1, beta2])
            print("Opt Params: Beta:", self.beta_params, "\tlenghtscale: ",self.char_length_scale, "\tVariance:",self.signal_variance)

        if (self.params_estimation):

            # Estimating Length scale itself
            x_max_value = None
            log_like_max =  -1* float("inf")

            # Data structure to create the starting points for the scipy.minimize method
            random_points_beta1 = []
            random_points_beta2 = []

            # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
            for dim in np.arange(self.number_of_dimensions):
                random_point_each_dim = np.random.uniform(self.len_scale_param_bounds[dim][0],
                                                               self.len_scale_param_bounds[dim][1],
                                                               self.number_of_restarts_likelihood). \
                    reshape(1, self.number_of_restarts_likelihood)
                random_points_beta1.append(random_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_beta1 = np.vstack(random_points_beta1)

            # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
            for dim in np.arange(self.number_of_dimensions):
                random_point_each_dim = np.random.uniform(self.len_scale_param_bounds[dim][0],
                                                               self.len_scale_param_bounds[dim][1],
                                                               self.number_of_restarts_likelihood). \
                    reshape(1, self.number_of_restarts_likelihood)
                random_points_beta2.append(random_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_beta2 = np.vstack(random_points_beta2)

            total_bounds = self.len_scale_param_bounds.copy()

            for ind in np.arange(self.number_of_restarts_likelihood):

                beta1 = random_points_beta1[0][ind]
                beta2 = random_points_beta2[0][ind]

                #
                # x = [0.05, 0.1]
                # value = self.optimize_log_marginal_likelihood_l_params(x)
                # print(value)

                init_points = np.append(beta1, beta2)
                print(ind+1, ". Initial values for a & b are ",beta1, beta2)
                maxima =  opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_l_params(x),
                                      init_points,
                                      method='L-BFGS-B',
                                      bounds=total_bounds
                                       # , options={'disp':True}
                                       , options={ 'maxfun': 2000, 'maxiter': 2000}
                                       )

                if not maxima['success']:
                    print ("Error occured : ", maxima['message'], "Log L: ", -1*maxima['fun'])
                    continue

                beta1_temp = maxima['x'][0]
                beta2_temp = maxima['x'][1]
                params = np.append(beta1_temp, beta2_temp)

                log_likelihood = -1* maxima['fun']

                print("Comparing", log_likelihood, log_like_max)
                if (log_likelihood > log_like_max):
                    print("New maximum log likelihood ", -1 * log_likelihood, " found for params ", params)
                    x_max_value = maxima
                    log_like_max = log_likelihood

            a = x_max_value['x'][0]
            b = x_max_value['x'][1]
            self.len_scale_params = np.array([a, b])

            print("Opt Params: ", self.len_scale_params)

        # compute the covariances between the test data points i.e K**
        K_xs_xs = self.computekernel(Xs, Xs)

        # Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
        L_xs_xs = np.linalg.cholesky(K_xs_xs + 1e-10 * np.eye(self.number_of_test_datapoints))

        # Sample 3 standard normals for our test points
        standard_normals = np.random.normal(size=(self.number_of_test_datapoints, 3))

        # multiply them by the square root of the covariance matrix
        f_prior = np.dot(L_xs_xs, standard_normals)

        plot_prior_params = {'plotnum': 'Fig 1_' + str(count),
                             'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                             'plotvalues': [[Xs, f_prior], [Xs, np.zeros(len(Xs)), 'b--', 'label=zero mean', 'lw1']],
                             'title': 'GP Prior Samples',
                             'file': 'GP_Prior'+ str(count),
                             'xlabel' : 'x',
                             'ylabel': 'output, f(x)'
                             }
        # self.plot_graph(plot_prior_params)

        mean, variance, factor1 = self.compute_mean_var(Xs, X, y)
        diag_variance = np.diag(variance)
        standard_deviation = np.sqrt(diag_variance)

        # compute posterior for the data points
        # newL = np.linalg.cholesky(K_xs_xs + 1e-10 * np.eye(self.number_of_test_datapoints) - np.dot(factor1.T, factor1))
        # f_post = mean.reshape(-1, 1) + np.dot(newL, np.random.normal(size=(self.number_of_test_datapoints, 3)))

        # plot_posterior_sample_params = {'plotnum': 'Fig 2_' + str(count),
        #                                 'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
        #                                 'plotvalues': [[X, y, 'r+', 'ms20'], [Xs, f_post]],
        #                                 'title': 'GP Posterior Samples',
        #                                 'file': 'GP_Posterior_Samples'+ str(count),
        #                                 'xlabel' : 'x',
        #                                 'ylabel': 'output, f(x)'
        #                                 }
        # self.plot_graph(plot_posterior_sample_params)

        plot_posterior_distr_params = {'plotnum': 'Fig 3_' + str(count),
                                       'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                                       'plotvalues': [[X, y, 'r+', 'ms20'], [Xs, ys, 'b-', 'label=True Fn'],
                                                      [Xs, mean, 'g--','label=Mean Fn','lw2']],
                                       'title': 'GP Posterior Distribution',
                                       'file': 'GP_Posterior_Distr'+ str(count),
                                       'gca_fill': [Xs.flat, mean - 2 * standard_deviation,
                                                    mean + 2 * standard_deviation] ,
                                       'xlabel': 'x',
                                       'ylabel': 'output, f(x)'
                                       }
        self.plot_graph(plot_posterior_distr_params)

        # len_scale_values = np.array([])
        # for each_point in Xs:
        #     len_scale_values = np.append(len_scale_values, self.len_scale_func(each_point))
        #
        # plot_len_scales = {'plotnum': 'Fig 4_' + str(count),
        #                    'axis': [linspacexmin,linspacexmax,0,1],
        #                    'plotvalues': [[Xs, len_scale_values,'g-', 'lw2']],
        #                    'title': 'Length scale functions l: ',
        #                    'file': 'Length Scale'+ str(count),
        #                    'xlabel': 'x',
        #                    'ylabel': 'Length scale function'
        #                    }
        # self.plot_graph(plot_len_scales)

        return mean


if __name__ == "__main__":

    timenow = datetime.datetime.now()
    print("\nStart time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    stamp = timenow.strftime("%H%M%S_%d%m%Y")
    f = open('console_output_' + str(stamp) + '.txt', 'w')
    original = sys.stdout
    sys.stdout = Custom_Print(sys.stdout, f)

    kernel_type = 0
    number_of_test_datapoints = 100
    noise = 0.0
    random_seed = 500
    kernel_char = "var_l"
    signal_variance = 1
    linspacexmin = 0
    linspacexmax = 10
    linspaceymin = -1.5
    linspaceymax = 3
    number_of_dimensions = 1
    params_estimation = False
    likelihood_estimation = True
    number_of_restarts_likelihood = 10
    number_of_observed_samples = 10


    a = 0.14
    b = 0.1
    len_scale_params = np.array([a, b])
    len_scale_param_bounds = [[0.01, 1] for nd in np.arange(len(len_scale_params))]
    lengthscale_bounds = [[0.5, 0.6]]
    signal_variance_bounds = [0.1, 1]

    beta_bounds = [[-1,1], [-1,1]]
    # X = np.array([2, 7]).reshape(2, 1)
    # X = np.array([2.5, 5, 7]).reshape(-1, 1)
    X = np.linspace(linspacexmin, linspacexmax, number_of_observed_samples).reshape(-1, 1)
    # number_of_observed_samples = len(X)

    # True function to be modelled
    # y = np.sin(X)
    y = np.exp(-(X - 2) ** 2) + np.exp(-(X - 6) ** 2 / 10) + 1 / (X ** 2 + 1)
    # y = (np.exp(-X) * np.sin(8 * np.pi * X)) + 1
    # y = (np.exp(-X) * np.sin(3 * np.pi * X)) + 0.3
    # y = (np.sin(X))/X
    # y = sinc_function(X)

    # test datapoints
    Xs = np.linspace(linspacexmin, linspacexmax, number_of_test_datapoints).reshape(-1, 1)
    # ys = np.sin(Xs)
    ys = np.exp(-(Xs - 2) ** 2) + np.exp(-(Xs - 6) ** 2 / 10) + 1 / (Xs ** 2 + 1)
    # ys = (np.exp(-Xs) * np.sin(8 * np.pi * Xs)) + 1
    # ys = (np.exp(-Xs) * np.sin(3 * np.pi * Xs)) + 0.3
    # ys = np.sin(Xs)/Xs
    # ys = sinc_function(Xs)

    kernel_char = 'fix_l'
    # kernel_char = 'var_l'
    # gaussianObject = GaussianProcess(kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
    #                                  linspacexmax, linspaceymin, linspaceymax, kernel_char, signal_variance,
    #                                  number_of_dimensions, number_of_observed_samples, X, y, params_estimation,
    #                                  len_scale_param_bounds,  number_of_restarts_likelihood, len_scale_params,
    #                                  beta_bounds, lengthscale_bounds, signal_variance_bounds )

    ##Other Params to reuse the code
    params_estimation = None
    len_scale_estimation = None
    len_scale_func_type = None
    len_weights_bounds = None
    weights_estimation = None
    multi_len_scales = None
    len_weights = None
    char_length_scale = 0.5
    len_scale_bounds = lengthscale_bounds
    bounds = [linspacexmin, linspacexmax]
    Xmin = linspacexmin
    Xmax = linspacexmax
    ymin = linspaceymin
    ymax = linspaceymax
    beta_params=[0.1,0.2]

    gaussianObject = GaussianProcess(kernel_type, params_estimation, len_scale_estimation, char_length_scale, len_scale_bounds,
                                     signal_variance, signal_variance_bounds, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                     linspacexmax,
                                     linspaceymin, linspaceymax, bounds, number_of_dimensions, number_of_observed_samples, kernel_char,
                                     len_scale_params, len_scale_param_bounds, len_scale_func_type, number_of_restarts_likelihood, Xmin,
                                     Xmax, ymin, ymax,
                                     len_weights_bounds, len_weights, weights_estimation, multi_len_scales,
                                    # added variables
                                     beta_bounds, beta_params)


    gaussianObject.gaussian_fit(X,y)

    number_of_GP_iterations = 1

    mean = None

    for i in np.arange(number_of_GP_iterations):

        if i == 0:
            # print("initial values: ", gaussianObject.len_scale_params)
            # mean = gaussianObject.runGaussian(i, Xs, ys, X, y)
            count = 10
            for each in ['fix_l']:
                kernel_char = each
                gaussianObject.kernel_char = each
                print("Kernel type: ",each)
                # if('var_l' == each):
                    # gaussianObject.params_estimation = Fal
                mean = gaussianObject.runGaussian(count, Xs, ys, X, y)
                count+=1

        if i == 1:
            print ("second iteration")

            newX=np.array([])
            newy=np.array([])

            for j in np.arange(30):
                rand = np.random.randint(1,100)
                # print(rand)
                newX = np.append(newX, Xs[rand])
                newy = np.append(newy, mean[rand])

            # # X = Xs[0:30]
            # # y = mean[0:30].reshape(-1,1)
            Xs = np.linspace(linspacexmin, linspacexmax, number_of_test_datapoints).reshape(-1, 1)
            # ys = (np.exp(-Xs) * np.sin(3 * np.pi * Xs)) + 0.3
            # ys = np.exp(-(Xs - 2) ** 2) + np.exp(-(Xs - 6) ** 2 / 10) + 1 / (Xs ** 2 + 1)
            # ys = np.sin(Xs)
            ys = mean
            gaussianObject.params_estimation = True
            gaussianObject.X = newX.reshape(-1,1)
            gaussianObject.y = newy.reshape(-1,1)
            gaussianObject.number_of_test_datapoints = number_of_test_datapoints
            gaussianObject.number_of_observed_samples = len(gaussianObject.X)
            gaussianObject.Xs = Xs
            gaussianObject.ys = ys
            gaussianObject.runGaussian(i, Xs, ys, newX.reshape(-1,1), newy.reshape(-1,1))

    timenow = datetime.datetime.now()
    print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    plt.show()


