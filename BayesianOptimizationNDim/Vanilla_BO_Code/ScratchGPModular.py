import numpy as np
import matplotlib.pyplot as plt
from AcquisitionFunction import AcquisitionFunction

# Class to handle Gaussian Process related tasks required for the Bayesian Optimization
class GaussianProcess:

    # Initializing the Gaussian Process object with the predefined GP Settings as specified by the user
    def __init__(self, kernel_type, params_estimation, char_length_scale, len_scale_bounds,
                 signal_variance, signal_variance_bounds,
                 number_of_test_datapoints, noise, random_seed, linspacexmin, linspacexmax,
                 linspaceymin, linspaceymax, bounds, number_of_dimensions, number_of_observed_samples):

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

    # Method to set the model used by the Gaussian Process
    def gaussian_fit(self,X, y):

        # Update the contents of X and y
        self.X = X

        # Normalisation and standardisation
        # y_mean = y.mean()
        # y_std = np.sqrt(y.var())
        # y = y-y_mean/y_std
        self.y = y

        # Recalculating L with updated length scale
        self.L_x_x = self.compute_l(X)
        print("L Recaluated for new data")

    # Define the kernel function to be used in the GP
    def computekernel(self, data_point1, data_point2):

        # Depending on the value specified by the user, appropriate kernel function is used in the Gaussian Process
        # Kernel_type = 0 represents Squared Exponential Kernel
        if self.kernel_type == 0:
            # print("SE Kernel")
            result = self.sq_exp_kernel(data_point1, data_point2, self.char_length_scale, self.signal_variance)

        # Kernel_type = 1 represents Rational Quadratic Function
        elif self.kernel_type == 1:
            print("RQF Kernel")
            # self.charac_length_scale = 1
            alpha = 0.1
            result = self.rational_quadratic_kernel(data_point1, data_point2, self.char_length_scale, alpha)

        # Kernel_type = 2 represents Exponential Kernel
        elif self.kernel_type == 2:
            print("EXP Kernel")
            # self.charac_length_scale = 0.1
            result = self.exp_kernel_function(data_point1, data_point2, self.char_length_scale)

        # Kernel_type = 3 represents the Periodic Kernel
        elif self.kernel_type == 3:
            print("Periodic Kernel")
            # self.charac_length_scale = 0.1
            result = self.periodic_kernel_function(data_point1, data_point2, self.char_length_scale)

        return result

    def sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        # Element wise squaring the vector of given length scales
        char_len_scale = np.array(char_len_scale) ** 2

        # construct a diagonal matrix with squared len_scale values
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



    def rational_quadratic_kernel(self, data_point1, data_point2, charac_length_scale, alpha):

        # Define Rational Quadratic Function
        # kernel formulation: k(x1,x2) = (1 + ((total_squared_distances(x1,x2) / (2.0 * (charac_length_scale ** 2) * alpha)))) ** (-alpha)
        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = (1 + ((total_squared_distances / (2.0 * (charac_length_scale ** 2) * alpha)))) ** (-alpha)
        return kernel_val

    def exp_kernel_function(self, data_point1, data_point2, charac_length_scale):

        # exponential covariance function , special case of matern with v = 1/2
        # kernel formulation: k(x1,x2) = exp{-(abs(x2 - x1) / charac_length_scale)}
        kernel_val = np.exp(-(abs(data_point2 - data_point1) / charac_length_scale))
        return kernel_val

    def periodic_kernel_function(self, data_point1, data_point2, charac_length_scale):

        # Periodic covariance function
        # kernel formulation: k(x1,x2) = exp{-2.0 * (sin(pi * (x2 - x1))) ** 2 * (1 / charac_length_scale ** 2)}
        kernel_val = np.exp(-2.0 * (np.sin(np.pi * (data_point2 - data_point1))) ** 2 * (1 / charac_length_scale ** 2))
        return kernel_val

    # Estimating kernel parameters (Hyperparameter tuning)
    def optimize_log_marginal_likelihood(self, input):
        # len(input)-1 items for lengthscales in different dimentsion
        init_charac_length_scale = np.array(input[: self.number_of_dimensions])
        # last element for signal variance
        signal_variance = input[len(input)-1]
        K_x_x = self.sq_exp_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        eye = 1e-6 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Inversion of kernel matrix via cholesky
        # Find L from K = L *L.T
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood

    def compute_l(self, X):
        # Use kernel function to find similarities between the observed samples
        K_x_x = self.computekernel(X, X)
        # print("x: ", X, "\nK: ",K_x_x )
        # Add some noise to avoid decay of eigen vectors (to avoid non positive definite kernel matrix)
        eye = 1e-6 * np.eye(len(X))
        # Inversion of kernel matrix via K
        L_x_x = np.linalg.cholesky(K_x_x + eye)
        self.L_x_x = L_x_x
        return L_x_x


    # Compute mean and variance required for the calculation of posteriors
    def compute_mean_var(self, Xs, X, y):

        # Use kernel function to find the covariance between unseen datapoints X* and the observed samples X
        K_x_xs = self.computekernel(X, Xs)
        factor1 = np.linalg.solve(self.L_x_x, K_x_xs)
        factor2 = np.linalg.solve(self.L_x_x, y)
        mean = np.dot(factor1.T, factor2)
        # mean = np.dot(factor1.T, factor2).flatten()

        # compute variance at test points
        # Use kernel function to find covariance between the unseen datapoints X*
        K_xs_xs = self.computekernel(Xs, Xs)
        variance = K_xs_xs - np.dot(factor1.T, factor1)

        return mean, variance, factor1

    # Method used to predict the mean and variance for the unseen data points X*
    def gaussian_predict(self, Xs):

        # compute the covariances between the unseen data points i.e K**
        K_xs_xs = self.computekernel(Xs, Xs)

        # Cholesky decomposition to find L for covariance matrix K i.e K = L*L.T
        L_xs_xs = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints))

        # Sample 3 standard normals for each of the unseen data points (function estimated using 3 samples drawn from R^D)
        standard_normals = np.random.normal(size=(self.number_of_test_datapoints, 3))

        # multiply them by the square root of the covariance matrix i.e., L
        f_prior = np.dot(L_xs_xs, standard_normals)

        # Compute mean and variance
        mean, variance, factor1 = self.compute_mean_var(Xs, self.X, self.y)
        diag_variance = np.diag(variance)
        standard_deviation = np.sqrt(diag_variance)

        # compute posteriors for the data points
        newL = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints) - np.dot(factor1.T, factor1))
        f_post = mean.reshape(-1, 1) + np.dot(newL, np.random.normal(size=(self.number_of_test_datapoints, 3)))

        return mean, diag_variance, f_prior, f_post



    def plot_graph(self, plot_params):

        ''' Generic method to plot 1D graphs according to the values passed
           Plot params can be specified as shown in the example below
           plot_params = {
                                'plotnum': 'plot_name/number',
                                'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin, self.linspaceymax],
                                'plotvalues': [[self.X, self.y, 'r+', 'ms15'], [Xs, ys, 'b-'], [Xs, mean, 'g--', 'lw2']],
                                              'title': 'GP Posterior Distribution with length scale L= ' + str(
                                                  self.char_length_scale),
                                              'file': 'GP_Posterior_Distr'+str(count),
                                              'gca_fill': [Xs.flat, (mean.flatten() - 2 * standard_deviation).reshape(-1,1).flat,
                                                           (mean.flatten() + 2 * standard_deviation).reshape(-1,1).flat]
                        }
        '''

        # Sets the plot name
        plt.figure(plot_params['plotnum'])

        # Clears the graph if any junk data is present from previous usages
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

        # Executed when there is a requirement to fill some region (to indicate the deviations or errors)
        if 'gca_fill' in plot_params.keys():

            # Depending on the parameters of the filling specified, appropriate block is called to render the graph
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

        plot_posterior_distr_params = {'plotnum': 'Posterior-'+str(count),
                                       'axis': [self.linspacexmin, self.linspacexmax, self.linspaceymin,
                                                self.linspaceymax],
                                       'plotvalues': [[self.X, self.y, 'r+', 'ms15'], [Xs, ys, 'b-'], [Xs, mean,
                                                                                                       'g--', 'lw2']],
                                       'title': 'GP Posterior Distribution', 'file': 'GP_Posterior_Distr'+str(count),
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
