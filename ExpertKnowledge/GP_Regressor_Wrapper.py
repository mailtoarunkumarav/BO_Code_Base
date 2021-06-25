import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


from Functions import FunctionHelper
from GP_Regressor import GaussianProcessRegressor
from HelperUtility.PrintHelper import PrintHelper as PH


class GPRegressorWrapper:

    def construct_gp_object(self, start_time, role, number_of_observed_samples, observations):

        # Multi Kernel for the initial experiments
        kernel_type = 'MKL'
        char_len_scale = 0.15
        number_of_test_datapoints = 500
        number_of_restarts_likelihood = 100
        noise = 0.0
        signal_variance = 1

        # # Square wave
        # linspacexmin = 0
        # linspacexmax = 4
        # linspaceymin = -1.5
        # linspaceymax = 1.5

        # Oscillator
        linspacexmin = 0
        linspacexmax = 8
        linspaceymin = 0
        linspaceymax = 2.5

        # Complicated Oscillator
        # linspacexmin = 0
        # linspacexmax = 30
        # linspaceymin = 0
        # linspaceymax = 2

        # Benchmark
        # linspacexmin = 0
        # linspacexmax = 10
        # linspaceymin = -1.5
        # linspaceymax = 3

        # Levy
        # linspacexmin = -10
        # linspacexmax = 10
        # linspaceymin = -16
        # linspaceymax = 0

        # Triangular wave
        # linspacexmin = 0
        # linspacexmax = 10
        # linspaceymin = -1.5
        # linspaceymax = 1.5

        # Chirpwave wave
        # linspacexmin = 0
        # linspacexmax = 20
        # linspaceymin = -3
        # linspaceymax = 3

        # Sinc Mixture
        # linspacexmin = -15
        # linspacexmax = 15
        # linspaceymin = -0.5
        # linspaceymax = 1.5

        # # Gaussian Mixture
        # linspacexmin = 0
        # linspacexmax = 15
        # linspaceymin = -0.5
        # linspaceymax = 1.5

        # Linear
        # linspacexmin = 0
        # linspacexmax = 2
        # linspaceymin = 0
        # linspaceymax = 0.5

        # Linear Sin Function
        # linspacexmin = 0
        # linspacexmax = 10
        # linspaceymin = 0
        # linspaceymax = 10

        # Gramacy Lee Function
        # linspacexmin = 0.5
        # linspacexmax = 2.5
        # linspaceymin = -1
        # linspaceymax = 1

        # Ackley Function
        # linspacexmin = -10
        # linspacexmax = 10
        # linspaceymin = -25
        # linspaceymax = 0.5

        number_of_dimensions = 1
        oned_bounds = [[linspacexmin, linspacexmax]]
        # sphere_bounds = [[linspacexmin, linspacexmax], [linspacexmin, linspacexmax]]
        # michalewicz2d_bounds = [[0, np.pi], [0, np.pi]]
        # random_bounds = [[0, 1], [1, 2]]
        # bounds = sphere_bounds
        # bounds = random_bounds
        bounds = oned_bounds

        Xmin = linspacexmin
        Xmax = linspacexmax
        ymax = linspaceymax
        ymin = linspaceymin

        lengthscale_bounds = [[0.1, 1]]
        signal_variance_bounds = [0.1, 1]
        true_func_type = "custom"
        fun_helper_obj = FunctionHelper(true_func_type)
        len_weights = [0.1, 0.3, 0.2]
        len_weights_bounds = [[0.1, 1] for i in range(3)]


        if role != "ai" and role != "baseline":

            if role == "GroundTruth":
                weight_params_estimation = True

            elif role == "HumanExpert":
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

            # Sinc
            # x_obs = np.linspace(-15, -5,  20)
            # x_obs = np.append(x_obs, np.linspace(5, 15, 20))
            # X= x_obs.reshape(-1, 1)

            # Gaussian
            # x_obs = np.linspace(0, 5,  20)
            # x_obs = np.append(x_obs, np.linspace(10, 15, 20))
            # X= x_obs.reshape(-1, 1)

            # #Linear
            # # X = np.linspace(linspacexmin, linspacexmax, 10).reshape(-1, 1)
            # x_obs = np.linspace(linspacexmin, 0.5,  5)
            # x_obs = np.append(x_obs, np.linspace(1.5, linspacexmax, 5))
            # X= x_obs.reshape(-1, 1)

            # Linear Sin Function
            # x_obs = np.linspace(linspacexmin, 3, 15)
            # x_obs = np.append(x_obs, np.linspace(7, linspacexmax, 15))
            # X = x_obs.reshape(-1, 1)

            # Commenting to adopt to Oscillator function multiple instances merged input domain
            # y = fun_helper_obj.get_true_func_value(X)

            y_arr = []
            for each_x in X:
                val = fun_helper_obj.get_true_func_value(each_x)
                y_arr.append(val)

            y = np.vstack(y_arr)

            X = np.divide((X - Xmin), (Xmax - Xmin))
            y = (y - ymin) / (ymax - ymin)

        elif role == "ai":
            weight_params_estimation = False
            X = observations["observations_X"]
            y = observations["observations_y"]

        elif role == "baseline":
            weight_params_estimation = True
            X = observations["observations_X"]
            y = observations["observations_y"]

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
        # ys = fun_helper_obj.get_true_func_value(Xs)

        ys_arr = []
        for each_xs in Xs:
            val_xs = fun_helper_obj.get_true_func_value(each_xs)
            ys_arr.append(val_xs)

        ys = np.vstack(ys_arr)
        # ys = sinc_function(Xs)

        Xs = np.divide((Xs - Xmin), (Xmax - Xmin))
        ys = (ys - ymin) / (ymax - ymin)

        gp_object = GaussianProcessRegressor(start_time, kernel_type, number_of_test_datapoints, noise, linspacexmin, linspacexmax,
                                             linspaceymin, linspaceymax, signal_variance, number_of_dimensions,
                                             number_of_observed_samples, X, y, number_of_restarts_likelihood, bounds, lengthscale_bounds,
                                             signal_variance_bounds, Xmin, Xmax, ymin, ymax, Xs, ys, char_len_scale, len_weights,
                                             len_weights_bounds, weight_params_estimation, fun_helper_obj)
        return gp_object


