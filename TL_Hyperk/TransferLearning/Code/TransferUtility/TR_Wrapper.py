import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys


import os
sys.path.append("..")
from HelperUtility.PrintHelper import PrintHelper as PH
sys.path.append("../GP_Regressor")
from Functions import FunctionHelper
from GP_Regression import GaussianProcess

# setting up the global parameters for plotting graphs i.e, graph size and suppress warning from multiple graphs
# being plotted
plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams["font.size"] = 12
plt.rcParams['figure.max_open_warning'] = 0
# np.seterr(divide='ignore', invalid='ignore')

# To fix the random number genration, currently not able, so as to retain the random selection of points
random_seed = 400
np.random.seed(random_seed)


# Class for starting Bayesian Optimization with the specified parameters
class TransferLearningWrapper:

    def initiate_transfer(self, gaussian_object):

        mean, variance, factor1 = gaussian_object.compute_mean_var(gaussian_object.Xs, gaussian_object.X, gaussian_object.y)

    def kernel_wrapper(self, start_time, input):

        kernel_type = 'SE'
        char_len_scale = 0.3
        number_of_test_datapoints = 500
        noise = 0.0
        random_seed = 500
        signal_variance = 1

        ## Benchmark function
        linspacexmin = 0
        linspacexmax = 10
        linspaceymin = -1.5
        linspaceymax = 3

        number_of_dimensions = 1
        number_of_observed_samples_src = 10
        number_of_observed_samples_tar = 20
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
        X_src = []

        # Generate specified (number of observed samples) random numbers for each dimension
        for dim in np.arange(number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                           number_of_observed_samples_src).reshape(1, number_of_observed_samples_src)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
        random_points = np.vstack(random_points)

        # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
        for sample_num in np.arange(number_of_observed_samples_src):
            array = []
            for dim_count in np.arange(number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            X_src.append(array)
        X_src = np.vstack(X_src)

        y_arr = []
        for each_x in X_src:
            val = func_helper_obj.get_true_func_value(each_x)
            y_arr.append(val)
        # y =  sinc_function(X)
        y_src = np.vstack(y_arr)

        # Normalising code commented
        X_src = np.divide((X_src - Xmin), (Xmax - Xmin))
        y_src = (y_src - ymin) / (ymax - ymin)

        # Commenting for regression data - Forestfire
        random_points = []
        X_tar = []

        # Generate specified (number of observed samples) random numbers for each dimension
        for dim in np.arange(number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                           number_of_observed_samples_tar).reshape(1, number_of_observed_samples_tar)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
        random_points = np.vstack(random_points)

        # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
        for sample_num in np.arange(number_of_observed_samples_tar):
            array = []
            for dim_count in np.arange(number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            X_tar.append(array)
        X_tar = np.vstack(X_tar)

        y_arr = []
        for each_x in X_tar:
            val = func_helper_obj.get_true_func_value(each_x)
            y_arr.append(val)
        # y =  sinc_function(X)
        y_tar = np.vstack(y_arr)

        # Normalising code commented
        X_tar = np.divide((X_tar - Xmin), (Xmax - Xmin))
        y_tar = (y_tar - ymin) / (ymax - ymin)

        random_points = []
        Xs_tar = []

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
            Xs_tar.append(array)
        Xs_tar = np.vstack(Xs_tar)
        # Obtain the values for the true function, so that true function can be plotted in the case of 1D currently

        # Comented to accomodate Oscillator function mergen input domain
        # ys = func_helper_obj.get_true_func_value(Xs)

        ys_arr = []
        for each_xs in Xs_tar:
            val_xs = func_helper_obj.get_true_func_value(each_xs)
            ys_arr.append(val_xs)
        ys_tar = np.vstack(ys_arr)
        # ys = sinc_function(Xs)

        # Normalising code commented
        Xs_tar = np.divide((Xs_tar - Xmin), (Xmax - Xmin))
        ys_tar = (ys_tar - ymin) / (ymax - ymin)


        gaussian_object_tar = GaussianProcess(kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                          linspacexmax, linspaceymin, linspaceymax, signal_variance, number_of_dimensions,
                                          number_of_observed_samples_tar, X_tar, y_tar, hyper_params_estimation,
                                          number_of_restarts_likelihood, lengthscale_bounds, signal_variance_bounds, func_helper_obj,
                                          Xmin, Xmax, ymin, ymax, Xs_tar, ys_tar, char_len_scale, len_weights, len_weights_bounds,
                                          weight_params_estimation)

        mean_src, variance_src, factor1 = gaussian_object_tar.compute_mean_var(X_tar, X_src, y_src)

        y_tar_diff = y_tar - mean_src.reshape(-1, 1)
        mean_corrected, variance_corrected, factor2 = gaussian_object_tar.compute_mean_var(X_src, X_tar, y_tar_diff)
        y_corrected = y_src + mean_corrected.reshape(-1, 1)



if __name__ == "__main__":
    timenow = datetime.datetime.now()
    stamp = timenow.strftime("%H%M%S_%d%m%Y")
    PH(os.getcwd())
    input = None
    trans_wrapper_obj = TransferLearningWrapper()
    trans_wrapper_obj.kernel_wrapper(stamp, input)
