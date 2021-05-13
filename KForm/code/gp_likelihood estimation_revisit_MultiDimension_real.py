import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import datetime
import sys
from Functions import FunctionHelper
from sklearn.model_selection import train_test_split
import pandas as pd
# kernel_type = 0
# number_of_test_datapoints = 20
np.random.seed(500)
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



class GaussianProcess:

    # Constructor
    def __init__(self, kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin, linspacexmax,
                 linspaceymin, linspaceymax, kernel_char, signal_variance, number_of_dimensions,
                 number_of_observed_samples, X, y, params_estimation, len_scale_param_bounds,
                 number_of_restarts_likelihood, len_scale_params, beta_bounds, lengthscale_bounds, signal_variance_bounds,
                 likelihood_estimation):

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
        self.beta_bounds = beta_bounds
        self.lengthscale_bounds = lengthscale_bounds
        self.signal_variance_bounds = signal_variance_bounds
        self.likelihood_estimation= likelihood_estimation

        # np.random.seed(random_seed)

    # Define Plot Prior
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


    # Define the kernel function
    def computekernel(self, data_point1, data_point2):


        if self.kernel_type == 0:
            # print("SE Kernel")
            #  result = self.sq_exp_kernel(data_point1, data_point2, self.charac_length_scale, self.signal_variance)
            result = self.free_kernel(data_point1, data_point2, self.charac_length_scale, self.signal_variance)
        return result


    def sq_exp_kernel_vanilla(self, data_point1, data_point2, charac_length_scale, signal_variance):

        # Define the SE kernel function
        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = np.exp(-(total_squared_distances * (1 / ((charac_length_scale**2) * 2.0))))
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
        random_vector1 = [0.2 for d in range(self.number_of_dimensions)]
        random_vector2 = [0.5 for d in range(self.number_of_dimensions)]

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
                # term2 = ( 2 * each_kernel_val - ((data_point1[i, :]) **2)- ((
                #               data_point2[j,:]) **2) - (np.square(rand_vector1))- (np.square(rand_vector2)))
                term2 = (2 * each_kernel_val - np.sum(np.square((data_point1[i, :]))) - np.sum(np.square((data_point2[j, :]))) - (np.sum(
                    np.square(rand_vector1))) - (np.sum(np.square(rand_vector2))))

                kernel_mat[i, j] = np.exp(0.5 * (1/(charac_length_scale**2)) * term2) * signal_variance

        return kernel_mat

    def optimize_log_marginal_likelihood_l(self, input):
        # 0 to n-1 elements represent the nth eleme

        self.beta_params = np.array(input[:2])
        self.signal_variance = input[len(input) - 1]
        # self.charac_length_scale = input[2: 2+self.number_of_dimensions][0]
        # K_x_x = self.sq_exp_kernel(self.X, self.X, self.charac_length_scale, signal_variance)
        init_charac_length_scale = input[2: 2+1]
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

        if (self.likelihood_estimation):

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

            # for dim in np.arange(self.number_of_dimensions):
            #     random_points_lengthscale_eachdim = np.random.uniform(self.lengthscale_bounds[dim][0],
            #                                               self.lengthscale_bounds[dim][1],
            #                                               self.number_of_restarts_likelihood). \
            #         reshape(1, self.number_of_restarts_likelihood)
            #     random_points_lengthscale.append(random_points_lengthscale_eachdim)
            #
            # # Vertically stack the arrays of randomly generated starting points as a matrix
            #     random_points_lengthscale = np.vstack(random_points_lengthscale)
            random_points_lengthscale_eachdim = np.random.uniform(self.lengthscale_bounds[0][0],
                                                          self.lengthscale_bounds[0][1],
                                                          self.number_of_restarts_likelihood). \
                    reshape(1, self.number_of_restarts_likelihood)
            random_points_lengthscale.append(random_points_lengthscale_eachdim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_lengthscale = np.vstack(random_points_lengthscale)

            variance_start_points = np.random.uniform(self.signal_variance_bounds[0],
                                                      self.signal_variance_bounds[1],
                                                      self.number_of_restarts_likelihood)

            total_bounds = self.beta_bounds.copy()
            total_bounds.append(self.lengthscale_bounds[0])
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
                maxima = opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_l(x),
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
            self.charac_length_scale = x_max_value['x'][2]
            self.signal_variance = x_max_value['x'][3]
            self.beta_params = np.array([beta1, beta2])
            print("Opt Params: Beta:", self.beta_params, "\tlenghtscale: ",self.charac_length_scale, "\tVariance:",self.signal_variance)

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


        # commenting for multi dimensional testing with regression
        # # compute posterior for the data points
        # # newL = np.linalg.cholesky(K_xs_xs + 1e-10 * np.eye(self.number_of_test_datapoints) - np.dot(factor1.T, factor1))
        # # f_post = mean.reshape(-1, 1) + np.dot(newL, np.random.normal(size=(self.number_of_test_datapoints, 3)))
        #
        # # plot_posterior_sample_params = {'plotnum': 'Fig 2_' + str(count),
        # #                                 'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
        # #                                 'plotvalues': [[X, y, 'r+', 'ms20'], [Xs, f_post]],
        # #                                 'title': 'GP Posterior Samples',
        # #                                 'file': 'GP_Posterior_Samples'+ str(count),
        # #                                 'xlabel' : 'x',
        # #                                 'ylabel': 'output, f(x)'
        # #                                 }
        # # self.plot_graph(plot_posterior_sample_params)
        #
        # plot_posterior_distr_params = {'plotnum': 'Fig 3_' + str(count),
        #                                'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
        #                                'plotvalues': [[X, y, 'r+', 'ms20'], [Xs, ys, 'b-', 'label=True Fn'],
        #                                               [Xs, mean, 'g--','label=Mean Fn','lw2']],
        #                                'title': 'GP Posterior Distribution',
        #                                'file': 'GP_Posterior_Distr'+ str(count),
        #                                'gca_fill': [Xs.flat, mean - 2 * standard_deviation,
        #                                             mean + 2 * standard_deviation] ,
        #                                'xlabel': 'x',
        #                                'ylabel': 'output, f(x)'
        #                                }
        # self.plot_graph(plot_posterior_distr_params)
        #
        # # len_scale_values = np.array([])
        # # for each_point in Xs:
        # #     len_scale_values = np.append(len_scale_values, self.len_scale_func(each_point))
        # #
        # # plot_len_scales = {'plotnum': 'Fig 4_' + str(count),
        # #                    'axis': [linspacexmin,linspacexmax,0,1],
        # #                    'plotvalues': [[Xs, len_scale_values,'g-', 'lw2']],
        # #                    'title': 'Length scale functions l: ',
        # #                    'file': 'Length Scale'+ str(count),
        # #                    'xlabel': 'x',
        # #                    'ylabel': 'Length scale function'
        # #                    }
        # # self.plot_graph(plot_len_scales)

        return mean


def sinc_function(X):
    value = X.copy()
    for i in range(X.shape[0]):
        if value[i] == 0:
            value[i] = 1
        else:
            value[i] = (np.sin(X[i]))/X[i]

    return value


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
    kernel_char = "ard"
    signal_variance = 1
    linspacexmin = 0
    linspacexmax = 10
    linspaceymin = -1.5
    linspaceymax = 3
    number_of_dimensions = 12
    number_of_observed_samples = 10
    params_estimation = False
    likelihood_estimation = True
    number_of_restarts_likelihood = 2
    sphere_bounds = [[linspacexmin, linspacexmax], [linspacexmin, linspacexmax]]
    michalewicz2d_bounds = [[0, np.pi], [0, np.pi]]
    # bounds = sphere_bounds
    bounds = michalewicz2d_bounds


    a = 0.14
    b = 0.1
    len_scale_params = np.array([a, b])
    len_scale_param_bounds = [[0.01, 1] for nd in np.arange(len(len_scale_params))]
    lengthscale_bounds = [[0.1, 1]]
    signal_variance_bounds = [0.1, 1]

    beta_bounds = [[-1,1], [-1,1]]

    true_func_type = "egg2d"
    func_helper_obj = FunctionHelper(true_func_type)

    # older versions of X and y
    # # X = np.array([2, 7]).reshape(2, 1)
    # # X = np.array([2.5, 5, 7]).reshape(-1, 1)
    # X = np.linspace(linspacexmin, linspacexmax, number_of_observed_samples).reshape(-1, 1)
    # # number_of_observed_samples = len(X)
    #
    # # True function to be modelled
    # # y = np.sin(X)
    # y = np.exp(-(X - 2) ** 2) + np.exp(-(X - 6) ** 2 / 10) + 1 / (X ** 2 + 1)
    # # y = (np.exp(-X) * np.sin(8 * np.pi * X)) + 1
    # # y = (np.exp(-X) * np.sin(3 * np.pi * X)) + 0.3
    # # y = (np.sin(X))/X
    # # y = sinc_function(X)
    #
    # # test datapoints
    # Xs = np.linspace(linspacexmin, linspacexmax, number_of_test_datapoints).reshape(-1, 1)
    # # ys = np.sin(Xs)
    # ys = np.exp(-(Xs - 2) ** 2) + np.exp(-(Xs - 6) ** 2 / 10) + 1 / (Xs ** 2 + 1)
    # # ys = (np.exp(-Xs) * np.sin(8 * np.pi * Xs)) + 1
    # # ys = (np.exp(-Xs) * np.sin(3 * np.pi * Xs)) + 0.3
    # # ys = np.sin(Xs)/Xs
    # # ys = sinc_function(Xs)

    # Commenting for regression data - Forestfire
    # random_points = []
    # X = []
    #
    # # Generate specified (number of observed samples) random numbers for each dimension
    # for dim in np.arange(number_of_dimensions):
    #     random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
    #                                                    number_of_observed_samples).reshape(1, number_of_observed_samples)
    #     random_points.append(random_data_point_each_dim)
    #
    # # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
    # random_points = np.vstack(random_points)
    #
    # # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
    # for sample_num in np.arange(number_of_observed_samples):
    #     array = []
    #     for dim_count in np.arange(number_of_dimensions):
    #         array.append(random_points[dim_count, sample_num])
    #     X.append(array)
    # X = np.vstack(X)
    # y = func_helper_obj.get_true_func_value(X)
    #
    # random_points = []
    # Xs = []
    #
    # # Generate specified (number of unseen data points) random numbers for each dimension
    # for dim in np.arange(number_of_dimensions):
    #     random_data_point_each_dim = np.linspace(bounds[dim][0], bounds[dim][1],
    #                                              number_of_test_datapoints).reshape(1,number_of_test_datapoints)
    #     random_points.append(random_data_point_each_dim)
    # random_points = np.vstack(random_points)
    #
    # # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
    # for sample_num in np.arange(number_of_test_datapoints):
    #     array = []
    #     for dim_count in np.arange(number_of_dimensions):
    #         array.append(random_points[dim_count, sample_num])
    #     Xs.append(array)
    # Xs = np.vstack(Xs)
    #
    # # Obtain the values for the true function, so that true function can be plotted in the case of 1D currently
    # ys = func_helper_obj.get_true_func_value(Xs)

    print("Working with Forest Fire Dataset")
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    # bcdata = pd.read_csv(url)
    bcdata = pd.read_csv("Dataset/forestfires.csv")
    bcdata.day[bcdata.day=='sun'] = 0
    bcdata.day[bcdata.day=='mon'] = 1
    bcdata.day[bcdata.day=='tue'] = 2
    bcdata.day[bcdata.day=='wed'] = 3
    bcdata.day[bcdata.day=='thu'] = 4
    bcdata.day[bcdata.day=='fri'] = 5
    bcdata.day[bcdata.day=='sat'] = 6

    bcdata.month[bcdata.month == 'jan'] = 0
    bcdata.month[bcdata.month == 'feb'] = 1
    bcdata.month[bcdata.month == 'mar'] = 2
    bcdata.month[bcdata.month == 'apr'] = 3
    bcdata.month[bcdata.month == 'may'] = 4
    bcdata.month[bcdata.month == 'jun'] = 5
    bcdata.month[bcdata.month == 'jul'] = 6
    bcdata.month[bcdata.month == 'aug'] = 7
    bcdata.month[bcdata.month == 'sep'] = 8
    bcdata.month[bcdata.month == 'oct'] = 9
    bcdata.month[bcdata.month == 'nov'] = 10
    bcdata.month[bcdata.month == 'dec'] = 11

    D = bcdata.drop(bcdata.columns[12], axis=1)

    f = bcdata.iloc[:, 12]

    newX_train, newXs, new_y, new_ys = train_test_split(D, f, test_size=0.80)
    X = newX_train.to_numpy()
    y = new_y.to_numpy()
    Xs = newXs.to_numpy()
    ys = new_ys.to_numpy()

    kernel_char = 'fix_l'
    # kernel_char = 'var_l'
    gaussianObject = GaussianProcess(kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                     linspacexmax, linspaceymin, linspaceymax, kernel_char, signal_variance,
                                     number_of_dimensions, number_of_observed_samples, X, y, params_estimation,
                                     len_scale_param_bounds,  number_of_restarts_likelihood, len_scale_params,
                                     beta_bounds, lengthscale_bounds, signal_variance_bounds, likelihood_estimation)

    number_of_GP_iterations = 1

    mean = None

    for i in np.arange(number_of_GP_iterations):

        if i == 0:
            count = 10
            for each in ['fix_l']:
                kernel_char = each
                gaussianObject.kernel_char = each
                print("Kernel type: ",each)
                mean = gaussianObject.runGaussian(count, Xs, ys, X, y)
                count+=1

    timenow = datetime.datetime.now()
    print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    plt.show()

