import numpy as np
import matplotlib.pyplot as plt

# kernel_type = 0 #
# number_of_test_datapoints = 20
# np.random.seed(500)
# noise = 0.0

# Class to handle the Gaussian Process related tasks required for the Bayesian Optimization
class GaussianProcess:

    # Initializing the Gaussian Process object with the predefined GP Settings as specified by the user
    def __init__(self, kernel_type, params_estimation, len_scale_estimation,char_length_scale, len_scale_bounds,
                 signal_variance, signal_variance_bounds,
                 number_of_test_datapoints, noise, random_seed, linspacexmin, linspacexmax,
                 linspaceymin, linspaceymax, bounds, number_of_dimensions, number_of_observed_samples, kernel_char,
                 len_scale_params, len_scale_param_bounds,len_scale_func_type, number_of_restarts_likelihood, Xmin, Xmax, ymin, ymax):

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
        #not required as we are generating random samples at each run
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
        self.y_original_list = None

    # Method to set the model used by the Gaussian Process
    def gaussian_fit(self,X, y):

        # Update the contents of X and y
        self.X = X
        # y_mean = y.mean()
        # y_std = np.sqrt(y.var())
        # y = y-y_mean/y_std
        self.y = y
        # Recalculating L with updated length scale
        self.L_x_x = self.compute_l(X)
        print("L Recaluated with new data")

    # Define the kernel function to be used in the GP
    def computekernel(self, data_point1, data_point2):

        # Depending on the setting specified by the user, appropriate kernel function is used in the Gaussian Process

        # Kernel_type = 0 represents the Squared Exponential Kernel
        if self.kernel_type == 0:
            # print("SE Kernel")
            result = self.sq_exp_kernel(data_point1, data_point2, self.char_length_scale, self.signal_variance)

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


        if( self.kernel_char == 'ard' or self.kernel_char == 'fix_l'):

            # if(self.params_estimation == True):
            #     print("ARD kernel is set for the computations")
            # else:
            # print("Fixed length kernel is set for the computations")
            return self.ard_sq_exp_kernel(data_point1, data_point2, char_len_scale, signal_variance)

        elif(self.kernel_char == 'var_l'):
            # print("Var kernel is set for the computations")
            return self.var_sq_exp_kernel(data_point1, data_point2, char_len_scale, signal_variance)

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
                each_kernel_val = (signal_variance**2) * (np.exp((-1 / 2.0) * final_product))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def len_scale_func_linear(self,data_point_value, len_scale):

        a = len_scale[0]
        # b = len_scale[1]
        b = 0.001
        value = a * data_point_value + b
        if value == 0:
            value = 1e-6

        return value

    def len_scale_func_quad(self, data_point_value, len_scale):

        a = len_scale[0]
        b = len_scale[1]
        c = len_scale[2]

        value = a * (data_point_value ** 2) + b * data_point_value + c
        if value == 0:
            value = 1e-6
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
                if(total_product < 0):
                    print("here", data_point1, data_point2)

                squared_diff = np.dot(difference, difference.T)
                each_kernel_val = (signal_variance ** 2) * np.sqrt(total_product) * (np.exp((-1) * squared_diff * total_sum))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def len_scale_func(self, data_point):

        # # Initializations for a and b
        # Linear Functions a.x + b
        # a = 0.085; b = 0.1;         # l : x + 1
        #a = -0.080; b = 0.9;         # l : x + 1
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
        signal_variance = input[len(input)-1]
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
            if (type == 'linear'):
                # total_len_scale_params.append(input[param_count:(param_count + 2)])
                total_len_scale_params.append(input[param_count:(param_count + 1)])
                # param_count += 2
                param_count += 1
            elif (type == 'quadratic'):
                total_len_scale_params.append(input[param_count:(param_count + 3)])
                param_count += 3


        self.len_scale_params = total_len_scale_params

        # Following parameters not used in any computations
        init_charac_length_scale = 0.1
        signal_variance = 1

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
        mean = np.dot(factor1.T, factor2)
        # mean = np.dot(factor1.T, factor2).flatten()

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
        newL = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints) - np.dot(factor1.T, factor1))
        f_post = mean.reshape(-1, 1) + np.dot(newL, np.random.normal(size=(self.number_of_test_datapoints, 3)))

        return mean, diag_variance, f_prior, f_post



    def plot_graph(self, plot_params):

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
        plot_prior_params = {'plotnum': 'Fig 1' ,
                             'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin, self.linspaceymax],
                             'plotvalues': [[Xs, f_prior], [Xs, np.zeros(len(Xs)), 'b--', 'lw2']],
                             'title': 'GP Prior Samples',
                             'file': 'GP_Prior'
                             }
        self.plot_graph(plot_prior_params)

    # Method used to plot posteriors with the specified f_post in the case of 1D problem
    def plot_posterior_samples(self, Xs, f_post):

        plot_posterior_sample_params = {'plotnum': 'Fig 2' ,
                                        'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,
                                                 self.linspaceymax],
                                        'plotvalues': [[self.X, self.y, 'r+', 'ms15'], [Xs, f_post]],
                                        'title': 'GP Posterior Samples',
                                        'file': 'GP_Posterior_Samples'
                                        }

        self.plot_graph(plot_posterior_sample_params)

    # Method used to plot the predictions in the case of 1D problem with the mean and standard deviations
    # and function's evaluations at observed samples as well as predictions from Gaussian Process
    def plot_posterior_predictions(self, count ,Xs, ys, mean, standard_deviation):

        plot_posterior_distr_params = {'plotnum': 'Fig 3-'+str(count),
                                       # 'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,self.linspaceymax],
                                       ## Normalizing code
                                       'axis': [0, 1, self.linspaceymin,self.linspaceymax],
                                       'plotvalues': [[self.X, self.y, 'r+', 'ms15'], [Xs, ys, 'b-'], [Xs, mean,
                                                                                                       'g--', 'lw2']],
                                       'title': 'GP Posterior Distr. with Spatially varying length scale'
                                                # + str(self.char_length_scale)
                                        ,'file': 'GP_Posterior_Distr'+str(count),
                                       'gca_fill': [Xs.flat, (mean.flatten() - 2 * standard_deviation).reshape(-1,1).flat,
                                                    (mean.flatten() + 2 * standard_deviation).reshape(-1,1).flat]
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
        plot_prior_params = {'plotnum': 'Fig 1_' ,
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
        plot_posterior_sample_params = {'plotnum': 'Fig 2_' ,
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
        kernel_val = signal_variance * (np.exp(-(total_squared_distances * (1 / ((charac_length_scale**2) * 2.0)))))
        return kernel_val

'''
#References

 # dist = numpy.linalg.norm(a - b)
        # total_squared_distances = np.linalg.norm(data_point1 - data_point2)

        # total_squared_distances = distance.euclidean(data_point1, data_point2)
        # print("total squared distances", total_squared_distances)
        # kernel_val = np.exp(-((total_squared_distances ** 2) * (1 / (charac_length_scale * 2.0))))

        # print(data_point1.shape, data_point2.shape)
        # print(data_point1, data_point2)
        # print( "dp**2", np.sum(data_point1 ** 2,1).reshape(-1,1))
        # print("product is ", 2 * np.dot(data_point1, data_point2.T))
        # print("sum:\n", np.sum(data_point2 ** 2,1).reshape(-1,1)+ np.sum(data_point2 ** 2,1).reshape(-1, 1) )

        # print(data_point1, data_point2)

        # print("dp1\n",data_point1, "\ndp2", data_point2)
        # print(type(data_point1), type(data_point2))
        # print("shapes", data_point1.shape, data_point2.shape)
        # summand1 = (data_point1**2).reshape(-1,1)
        # print("summand:", summand1)
        # sq_sums = summand1 + data_point2 ** 2
        # # sq_sums = np.dot(data_point1, data_point1.T) + np.dot(data_point2, data_point2.T)
        # product = 2 * np.dot(data_point1, data_point2.T)
        # print(sq_sums)
        # total_squared_distances =  sq_sums - product
        # kernel_val = np.exp(-(total_squared_distances * (1 / (charac_length_scale * 2.0))))

        # print ("kernel: ", kernel_val)

    def sq_exp_kernel_old(self, data_point1, data_point2, charac_length_scale):
        ## Simple compute mechanism but inefficient
        # asquared = data_point1**2
        # bsquared = data_point2**2
        # twoab = 2 * np.dot(data_point1,data_point2.T)
        # matrix = []
        # for ai in asquared:
        #     array=[]
        #     for bi in bsquared:
        #         array.extend(ai+bi)
        #     matrix.append(array)
        #     array = []
        # sq_dist = matrix - twoab
        # value = np.exp(-(sq_dist*(1/(charac_length_scale*2.0))))
        # print (value,"****\n\n")

        # Define the SE kernel function

        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = np.exp(-(total_squared_distances * (1 / (charac_length_scale * 2.0))))
        # print (kernel_val)
        return kernel_val


 
'''
