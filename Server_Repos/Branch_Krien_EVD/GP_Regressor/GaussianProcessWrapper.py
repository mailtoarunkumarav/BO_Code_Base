from GP_Regression import GaussianProcess
from Functions import FunctionHelper
import numpy as np


class GaussianProcessWrapper:
    def __init__(self):
        self.gaussian_object = None

    def construct_gp_regressor(self):

        kernel_type = 'SE'
        char_len_scale = 0.3
        number_of_test_datapoints = 500
        noise = 0.0
        random_seed = 500
        signal_variance = 1

        # Benchmark function
        # linspacexmin = 0
        # linspacexmax = 10
        # linspaceymin = -1.5
        # linspaceymax = 3

        # Levy funciton
        # linspacexmin = -10
        # linspacexmax = 10
        # linspaceymin = -16
        # linspaceymax = 0

        # # Complicated Oscillator
        # linspacexmin = 0
        # linspacexmax = 30
        # linspaceymin = 0
        # linspaceymax = 2

        # Oscillator
        # linspacexmin = 0
        # linspacexmax = 8
        # linspaceymin = 0
        # linspaceymax = 2.5

        # Square wave
        # linspacexmin = 0
        # linspacexmax = 4
        # linspaceymin = -1.5
        # linspaceymax = 1.5

        # Triangular wave
        # linspacexmin = 0
        # linspacexmax = 10
        # linspaceymin = -1.5
        # linspaceymax = 1.5

        # Square wave
        linspacexmin = 0
        linspacexmax = 20
        linspaceymin = -3
        linspaceymax = 3


        number_of_dimensions = 1
        number_of_observed_samples = 20
        hyper_params_estimation = True
        number_of_restarts_likelihood = 100
        oned_bounds = [[linspacexmin, linspacexmax]]
        sphere_bounds = [[linspacexmin, linspacexmax], [linspacexmin, linspacexmax]]
        michalewicz2d_bounds = [[0, np.pi], [0, np.pi]]
        random_bounds = [[0, 1], [1, 2]]
        # bounds = sphere_bounds
        bounds = oned_bounds
        # bounds = random_bounds

        Xmin = linspacexmin
        Xmax = linspacexmax
        ymin = linspaceymin
        ymax = linspaceymax

        a = 0.14
        b = 0.1
        lengthscale_bounds = [[0.1, 10]]
        signal_variance_bounds = [0.1, 10]
        true_func_type = "custom"
        func_helper_obj = FunctionHelper(true_func_type)
        len_weights = [0.1, 0.3, 0.2]
        len_weights_bounds = [[0.1, 5] for i in range(3)]
        weight_params_estimation = False

        # Commenting for regression data - Forestfire
        random_points = []
        X = []

        # Generate specified (number of observed samples) random numbers for each dimension
        for dim in np.arange(number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
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
        print(X)
        # X = np.linspace(linspacexmin, linspacexmax, 10).reshape(-1, 1)

        # Commenting to adopt to Oscillator function multiple instances merged input domain
        # y = fun_helper_obj.get_true_func_value(X)
        y_arr = []
        for each_x in X:
            val = func_helper_obj.get_true_func_value(each_x)
            y_arr.append(val)
        # y =  sinc_function(X)
        y = np.vstack(y_arr)

        # Normalising code commented
        X = np.divide((X - Xmin), (Xmax - Xmin))
        y = (y - ymin) / (ymax - ymin)

        random_points = []
        Xs = []

        # Generate specified (number of unseen data points) random numbers for each dimension
        for dim in np.arange(number_of_dimensions):
            random_data_point_each_dim = np.linspace(bounds[dim][0], bounds[dim][1],
                                                     number_of_test_datapoints).reshape(1, number_of_test_datapoints)
            random_points.append(random_data_point_each_dim)
        random_points = np.vstack(random_points)

        # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
        for sample_num in np.arange(number_of_test_datapoints):
            array = []
            for dim_count in np.arange(number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            Xs.append(array)
        Xs = np.vstack(Xs)
        # Obtain the values for the true function, so that true function can be plotted in the case of 1D currently

        # Comented to accomodate Oscillator function mergen input domain
        # ys = func_helper_obj.get_true_func_value(Xs)

        ys_arr = []
        for each_xs in Xs:
            val_xs = func_helper_obj.get_true_func_value(each_xs)
            ys_arr.append(val_xs)
        ys = np.vstack(ys_arr)
        # ys = sinc_function(Xs)


        #Normalising code commented
        Xs = np.divide((Xs - Xmin), (Xmax - Xmin))
        ys = (ys - ymin) / (ymax - ymin)

        gaussian_object = GaussianProcess(kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                         linspacexmax, linspaceymin, linspaceymax, signal_variance,
                                         number_of_dimensions, number_of_observed_samples, X, y, hyper_params_estimation,
                                         number_of_restarts_likelihood, lengthscale_bounds, signal_variance_bounds, func_helper_obj,
                                         Xmin, Xmax, ymin, ymax, Xs, ys, char_len_scale, len_weights, len_weights_bounds,
                                         weight_params_estimation)
        self.gaussian_object = gaussian_object
        return

    def compute_likelihood_for_kernel(self, kernel_type, observations_kernel, optional_hypergp_obj):
        count = 1
        likelihood = self.gaussian_object.runGaussian(count, kernel_type, observations_kernel, optional_hypergp_obj)
        return likelihood

    def compute_posterior_distribution(self, kernel_type, observations_kernel, optional_hypergp_obj, msg):
        self.gaussian_object.plot_posterior_distribution(kernel_type, observations_kernel, optional_hypergp_obj, msg)