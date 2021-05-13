import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import datetime

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 20
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)

# kernel_type = 0
# number_of_test_datapoints = 20
np.random.seed(300)
# noise = 0.0


class GaussianProcess:

    # Constructor
    def __init__(self, kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin, linspacexmax,
                 linspaceymin, linspaceymax, kernel_char, signal_variance, number_of_dimensions,
                 number_of_observed_samples, X, y, params_estimation, len_scale_param_bounds,
                 number_of_restarts_likelihood, len_scale_params):

        self.kernel_type = kernel_type
        self.number_of_test_datapoints = number_of_test_datapoints
        self.noise = noise
        self.linspacexmin = linspacexmin
        self.linspacexmax = linspacexmax
        self.linspaceymin = linspaceymin
        self.linspaceymax = linspaceymax
        self.kernel_char = kernel_char
        self.signal_variance = signal_variance
        self.number_of_dimensions = number_of_dimensions
        self.number_of_observed_samples = number_of_observed_samples
        self.X = X
        self.y = y
        self.params_estimation = params_estimation
        self.len_scale_param_bounds = len_scale_param_bounds
        self.number_of_restarts_likelihood = number_of_restarts_likelihood
        self.len_scale_params = len_scale_params

        # np.random.seed(random_seed)

    # Define Plot Prior
    def plot_graph(self, plot_params):
        plt.figure(plot_params['plotnum'])
        plt.clf()
        for eachplot in plot_params['plotvalues']:
            if (len(eachplot) == 2):
                plt.plot(eachplot[0], eachplot[1],  lw =3)
            elif (len(eachplot) == 3):
                plt.plot(eachplot[0], eachplot[1])
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

        # Adding to have subfigure name printed at bottom
        if(kernel_char=="fix_l"):
            title = "(b)"
            plt.title(title, y=-0.24)
        else:
            title = "(a)"
            plt.title(title, y=-0.24)

        # plt.title(plot_params['title'])
        plt.xlabel(plot_params['xlabel'])
        plt.ylabel(plot_params['ylabel'])
        plt.ylim(-3,3.4)
        plt.legend(fontsize='small',loc = 4)
        # plt.use_sticky_edges = False
        # # plt.autoscale(tight=True)
        # # plt.autoscale(tight=True)
        plt.savefig(plot_params['file'],pad_inches=0, bbox_inches='tight')


    def plot_graph_len(self, plot_params):
        plt.figure(plot_params['plotnum'])
        plt.clf()
        for eachplot in plot_params['plotvalues']:
            if (len(eachplot) == 2):
                plt.plot(eachplot[0], eachplot[1],  lw =3.5)
            elif (len(eachplot) == 3):
                plt.plot(eachplot[0], eachplot[1])
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
        plt.autoscale(tight=True)
        plt.ylim(0, 1.2)

        plt.savefig(plot_params['file'],pad_inches=0, bbox_inches='tight')


    # Define the kernel function
    def computekernel(self, data_point1, data_point2):

        if self.kernel_type == 0:
            # print("SE Kernel")
            self.charac_length_scale = 0.2
            result = self.sq_exp_kernel(data_point1, data_point2, self.charac_length_scale, self.signal_variance)

        elif self.kernel_type == 1:
            print("RQF Kernel")
            self.charac_length_scale = 1
            alpha = 0.1
            result = self.rational_quadratic_kernel(data_point1, data_point2, self.charac_length_scale, alpha)

        elif self.kernel_type == 2:
            print("EXP Kernel")
            self.charac_length_scale = 0.1
            result = self.exp_kernel_function(data_point1, data_point2, self.charac_length_scale)

        elif self.kernel_type == 3:
            print("Periodic Kernel")
            self.charac_length_scale = 0.1
            result = self.periodic_kernel_function(data_point1, data_point2, self.charac_length_scale)

        return result

    def sq_exp_kernel_vanilla(self, data_point1, data_point2, charac_length_scale, signal_variance):

        # Define the SE kernel function
        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = np.exp(-(total_squared_distances * (1 / (charac_length_scale * 2.0))))
        # print (kernel_val)
        return kernel_val

    def sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        if( self.kernel_char == 'ard' or self.kernel_char == 'fix_l'):

            return self.ard_sq_exp_kernel(data_point1, data_point2, [char_len_scale], signal_variance)

        elif(self.kernel_char == 'var_l'):
            # print("Var kernel is set for the computations")
            return self.var_sq_exp_kernel(data_point1, data_point2, char_len_scale, signal_variance)

    def ard_sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

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

    def var_sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):

                len_scale_vector_datapoint1 = self.len_scale_func(data_point1[i, :])
                len_scale_vector_datapoint2 = self.len_scale_func(data_point2[j, :])
                difference = ((data_point1[i, :] - data_point2[j, :]))

                total_product = 1
                total_sum = 0

                for k in np.arange(len(len_scale_vector_datapoint1)):
                    denominator = len_scale_vector_datapoint1[k] ** 2 + len_scale_vector_datapoint2[k] ** 2
                    total_product *= (2 * len_scale_vector_datapoint1[k] * len_scale_vector_datapoint2[k]) / denominator
                    total_sum += 1 / denominator

                squared_diff = np.dot(difference, difference.T)
                prod_term = (signal_variance ** 2) * np.sqrt(total_product)
                sum_term  = (np.exp((-1) * squared_diff * total_sum))
                each_kernel_val = prod_term * sum_term
                kernel_mat[i, j] = each_kernel_val

        return kernel_mat

    # Estimating kernel parameters
    def optimize_log_marginal_likelihood_l_params(self, input):
        # 0 to n-1 elements represent the nth eleme

        self.len_scale_params = np.array(input)

        # Following parameters not used in any computations
        init_charac_length_scale = 0.1
        signal_variance = 1

        K_x_x = self.sq_exp_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        eye = 1e-10 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        det = np.linalg.det(Knoise)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(det))
        return log_marginal_likelihood

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


    def kernel_params (self, data_point1, data_point2, char_len_scale, signal_variance):
        prod_array = np.array([])
        sum_array = np.array([])
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):

                len_scale_vector_datapoint1 = self.len_scale_func(data_point1[i, :])
                len_scale_vector_datapoint2 = self.len_scale_func(data_point2[j, :])
                difference = ((data_point1[i, :] - data_point2[j, :]))

                total_product = 1
                total_sum = 0

                for k in np.arange(len(len_scale_vector_datapoint1)):
                    denominator = len_scale_vector_datapoint1[k] ** 2 + len_scale_vector_datapoint2[k] ** 2
                    total_product *= (2 * len_scale_vector_datapoint1[k] * len_scale_vector_datapoint2[k]) / denominator
                    total_sum += 1 / denominator

                squared_diff = np.dot(difference, difference.T)
                prod_term = (signal_variance ** 2) * np.sqrt(total_product)
                prod_array = np.append(prod_array,prod_term)
                sum_term = (np.exp((-1) * squared_diff * total_sum))
                sum_array = np.append(sum_array, sum_term)
                each_kernel_val = prod_term * sum_term
                kernel_mat[i, j] = each_kernel_val

        return prod_array, sum_array, kernel_mat

    def len_scale_func(self, data_point):

        a = self.len_scale_params[0]
        b = self.len_scale_params[1]
        c = 0

        a = 0.2
        b = 0.1

        # len_scale_weights = np.zeros(data_point.shape)
        len_scale_weights = np.array([])
        len_scale_values = np.array([])
        data_point_values = np.array([])

        for dim_count in np.arange(len(data_point)):


            # # Linear calculations
            len_scale_weights = np.append(len_scale_weights, a)
            data_point_values = np.append(data_point_values, data_point[0])
            value = np.dot(len_scale_weights.T, data_point) + b

            # Quadratic calculations
            # len_scale_weights = np.append(len_scale_weights,a)
            # len_scale_weights = np.append(len_scale_weights,b)
            # data_point_values = np.append(data_point_values, data_point[0] ** 2)
            # data_point_values = np.append(data_point_values, data_point[0])
            # value = np.dot(len_scale_weights.T, data_point_values) + c

            # # Gaussian
            # mean = 2
            # std_dev = 0.8
            # bias = 0
            # exp_term = np.exp((-0.5) * (((data_point_values - mean) / std_dev) ** 2))
            # # pre_term = -1 / np.sqrt(2 * np.pi * (std_dev ** 2))
            # pre_term = 1
            # value = pre_term * exp_term + bias

            # Inverted Gaussian
            # mean = 2
            # std_dev = 0.8
            # bias = 1.1
            # exp_term = np.exp((-0.5) * (((data_point_values - mean) / std_dev) ** 2))
            # # pre_term = -1 / np.sqrt(2 * np.pi * (std_dev ** 2))
            # pre_term = -1
            # value = pre_term * exp_term + bias


            #Logistic
            # ### # Logistic function for length scale functions
            # value = (1/(1.25+np.exp(5-data_point_values)))+0.1

            if value == 0 :
                value = 1e-6
            len_scale_values = np.append(len_scale_values, value)

        return len_scale_values


    def rational_quadratic_kernel(self, data_point1, data_point2, charac_length_scale, alpha):

        # Define Rational Quadratic Function
        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = (1 + ((total_squared_distances / (2.0 * (charac_length_scale ** 2) * alpha)))) ** (-alpha)
        print(kernel_val)
        return kernel_val

    def exp_kernel_function(self, data_point1, data_point2, charac_length_scale):

        # exponential covariance function , special case of matern with v = 1/2
        kernel_val = np.exp(-(abs(data_point2 - data_point1) / charac_length_scale))
        print(kernel_val.shape)
        return kernel_val

    def periodic_kernel_function(self, data_point1, data_point2, charac_length_scale):

        # Periodic covariance function
        kernel_val = np.exp(-2.0 * (np.sin(np.pi * (data_point2 - data_point1))) ** 2 * (1 / charac_length_scale ** 2))
        print(kernel_val.shape)
        return kernel_val

    def compute_mean_var(self, Xs, X, y):

        # Apply the kernel function to our training points
        K_x_x = self.computekernel(X, X)

        eye = 1e-10 * np.eye(len(X))
        L_x_x = np.linalg.cholesky(K_x_x + eye)

        K_x_xs = self.computekernel(X, Xs)
        factor1 = np.linalg.solve(L_x_x, K_x_xs)
        factor2 = np.linalg.solve(L_x_x, y)
        mean = np.dot(factor1.T, factor2).flatten()

        K_xs_xs = self.computekernel(Xs, Xs)
        variance = K_xs_xs - np.dot(factor1.T, factor1)

        return mean, variance, factor1

    def runGaussian(self, count, Xs, ys, X, y):

        print("!!!!!!!!!!Gaussian Process: Iteration", count, " Started!!!!!!!!!")

        if (self.params_estimation):
            print("estimating params")
            # Estimating Length scale itself
            x_max_value = None
            log_like_max =  -1* float("inf")

            # Data structure to create the starting points for the scipy.minimize method
            random_points_a = []
            random_points_b = []

            # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
            for dim in np.arange(self.number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(self.len_scale_param_bounds[dim][0],
                                                               self.len_scale_param_bounds[dim][1],
                                                               self.number_of_restarts_likelihood). \
                    reshape(1, self.number_of_restarts_likelihood)
                random_points_a.append(random_data_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_a = np.vstack(random_points_a)

            # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
            for dim in np.arange(self.number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(self.len_scale_param_bounds[dim][0],
                                                               self.len_scale_param_bounds[dim][1],
                                                               self.number_of_restarts_likelihood). \
                    reshape(1, self.number_of_restarts_likelihood)
                random_points_b.append(random_data_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_b = np.vstack(random_points_b)

            total_bounds = self.len_scale_param_bounds.copy()

            for ind in np.arange(self.number_of_restarts_likelihood):

                param_a = random_points_a[0][ind]
                param_b = random_points_b[0][ind]

                #
                # x = [0.05, 0.1]
                # value = self.optimize_log_marginal_likelihood_l_params(x)
                # print(value)

                init_points = np.append(param_a, param_b)
                print(ind+1, ". Initial values for a & b are ",param_a, param_b)
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

                a_temp = maxima['x'][0]
                b_temp = maxima['x'][1]
                params = np.append(a_temp, b_temp)

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


        #Plotting here

        # Xs = np.linspace(0,1, 100).reshape(100,1)

        plot_prior_params = {'plotnum': 'Fig 1_' + str(count),
                             # 'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                             'axis': [0,1,-2,2],
                             'plotvalues': [[np.linspace(0,1, 100).reshape(100,1), f_prior]],
                             'title': 'GP Prior Samples',
                             'file': 'prior_log_l.pdf',
                             'xlabel' : 'X',
                             'ylabel': 'output, f(X)'
                             }
        self.plot_graph(plot_prior_params)

        len_scale_values = np.array([])
        for each_point in Xs:
            len_scale_values = np.append(len_scale_values, self.len_scale_func(each_point))

        plot_len_scales = {'plotnum': 'Fig 4_' + str(count),
                           'axis': [0,4,-1,1],
                           'plotvalues': [[np.linspace(0,1, 100).reshape(100,1), len_scale_values,'g-', 'lw3.5']],
                           'title': 'Logistic',
                           'file': 'log_l.pdf',
                           'xlabel': 'X',
                           'ylabel': 'length scale function l(X)'
                           }

        # self.plot_graph_len(plot_len_scales)

        # plt.show()
        # exit(0)

        mean, variance, factor1 = self.compute_mean_var(Xs, X, y)
        diag_variance = np.diag(variance)
        standard_deviation = np.sqrt(diag_variance)

        # compute posterior for the data points
        newL = np.linalg.cholesky(K_xs_xs + 1e-10 * np.eye(self.number_of_test_datapoints) - np.dot(factor1.T, factor1))
        f_post = mean.reshape(-1, 1) + np.dot(newL, np.random.normal(size=(self.number_of_test_datapoints, 3)))

        plot_posterior_sample_params = {'plotnum': 'Fig 2_' + str(count),
                                        'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                                        'plotvalues': [[X, y, 'r+', 'ms20'], [Xs, f_post]],
                                        'title': 'GP Posterior Samples',
                                        'file': 'GP_Posterior_Samples'+ str(count),
                                        'xlabel' : 'x',
                                        'ylabel': 'output, f(x)'
                                        }
        # self.plot_graph(plot_posterior_sample_params)

        # if (kernel_char == "var_l"):
        #     title = "Posterior for Spatially Varying Kernel"
        # elif (kernel_char == "fix_l"):
        #     title = "Posterior for Fixed SE Kernel"

        plot_posterior_distr_params = {'plotnum': 'Fig 3_' + str(count),
                                       'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                                       'plotvalues': [[X, y, 'r+', 'ms20'], [Xs, ys, 'b-', 'label=True Function','lw2'],
                                                      [Xs, mean, 'g--','label=Mean Function','lw3']],
                                       'title': "",
                                       'file': 'GP_Posterior_Distr'+ kernel_char+str(count)+".pdf",
                                       'gca_fill': [Xs.flat, mean - 2 * standard_deviation,
                                                    mean + 2 * standard_deviation] ,
                                       'xlabel': 'x',
                                       'ylabel': 'output, f(x)'
                                       }
        self.plot_graph(plot_posterior_distr_params)



        return mean



if __name__ == "__main__":

    timenow = datetime.datetime.now()
    print("\nStart time: ", timenow.strftime("%H%M%S_%d%m%Y"))

    kernel_type = 0
    number_of_test_datapoints = 100
    noise = 0.0
    random_seed = 500
    kernel_char = "var_l"
    signal_variance = 1.5
    linspacexmin = 0
    linspacexmax = 4
    linspaceymin = -3
    linspaceymax = 3
    number_of_dimensions = 1
    params_estimation = False
    number_of_restarts_likelihood = 15

    a = 0.14
    b = 0.1
    len_scale_params = np.array([a, b])
    len_scale_param_bounds = [[0.01, 1] for nd in np.arange(len(len_scale_params))]

    # X = np.array([2, 7]).reshape(2, 1)
    # X = np.array([2.5, 5, 7]).reshape(-1, 1)
    X = np.linspace(linspacexmin, linspacexmax, 10).reshape(-1, 1)
    number_of_observed_samples = len(X)

    # True function to be modelled
    # y = np.sin(X)
    # y = np.exp(-(X - 2) ** 2) + np.exp(-(X - 6) ** 2 / 10) + 1 / (X ** 2 + 1)
    y = (np.exp(-X) * np.sin(8 * np.pi * X)) + 1
    # y = (np.exp(-X) * np.sin(3 * np.pi * X)) + 0.3

    # test datapoints
    Xs = np.linspace(linspacexmin, linspacexmax, number_of_test_datapoints).reshape(-1, 1)
    # ys = np.sin(Xs)
    # ys = np.exp(-(Xs - 2) ** 2) + np.exp(-(Xs - 6) ** 2 / 10) + 1 / (Xs ** 2 + 1)
    ys = (np.exp(-Xs) * np.sin(8 * np.pi * Xs)) + 1
    # ys = (np.exp(-Xs) * np.sin(3 * np.pi * Xs)) + 0.3

    # kernel_char = 'fix_l'
    kernel_char = 'var_l'
    gaussianObject = GaussianProcess(kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                     linspacexmax, linspaceymin, linspaceymax, kernel_char, signal_variance,
                                     number_of_dimensions, number_of_observed_samples, X, y, params_estimation,
                                     len_scale_param_bounds,  number_of_restarts_likelihood, len_scale_params)

    number_of_GP_iterations = 1

    mean = None

    for i in np.arange(number_of_GP_iterations):

        if i == 0:
            # print("initial values: ", gaussianObject.len_scale_params)
            # mean = gaussianObject.runGaussian(i, Xs, ys, X, y)
            count = 10
            for each in ['fix_l' ,'var_l']:
            # for each in ['var_l']:
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
