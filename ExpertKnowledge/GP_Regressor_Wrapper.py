import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


from Functions import FunctionHelper
from GP_Regressor import GaussianProcessRegressor
from HelperUtility.PrintHelper import PrintHelper as PH


class GPRegressorWrapper:

    def construct_gp_object(self, start_time, role, number_of_random_observed_samples, function_type, observations, estimated_kernel):

        # Multi Kernel for the initial experiments

        kernel_type = 'SE'

        # if role == "GroundTruth" or role == "HumanExpert":
        #     kernel_type = 'LIN'
        # elif role == "ai" or role == "baseline":
        #     kernel_type = 'SE'
        #     # kernel_type = 'MKL'
        #     # kernel_type = estimated_kernel

        char_len_scale = 0.35
        number_of_test_datapoints = 500
        number_of_restarts_likelihood = 100
        noise = 0.0
        signal_variance = 1

        if function_type == "OSC1D":
            linspacexmin = 0
            linspacexmax = 8
            linspaceymin = 0
            linspaceymax = 2.5

        elif function_type == "GCL1D":
            linspacexmin = 0.5
            linspacexmax = 2.5
            linspaceymin = -5
            linspaceymax = 1

        elif function_type == "ACK1D":
            linspacexmin = -10
            linspacexmax = 10
            linspaceymin = -25
            linspaceymax = 0.5

        elif function_type == "BEN1D":
            # linspacexmin = 0
            # linspacexmax = 10
            # linspaceymin = -1.5
            # linspaceymax = 3

            linspacexmin = 0
            linspacexmax = 10
            linspaceymin = -0.9
            linspaceymax = 0.9

        elif function_type == "LIN1D":
            linspacexmin = 0
            linspacexmax = 4
            linspaceymin = -5
            linspaceymax = 5

        elif function_type == "LINSIN1D":
            linspacexmin = 0
            linspacexmax = 4
            linspaceymin = -2.5
            linspaceymax = 2.5

        # # Square wave
        # linspacexmin = 0
        #         # linspacexmax = 4
        #         # linspaceymin = -1.5
        #         # linspaceymax = 1.5
        #
        #         # Oscillator
        #         # linspacexmin = 0
        #         # linspacexmax = 8
        #         # linspaceymin = 0
        #         # linspaceymax = 2.5
        #
        #         # Complicated Oscillator
        #         # linspacexmin = 0
        #         # linspacexmax = 30
        #         # linspaceymin = 0
        #         # linspaceymax = 2
        #
        #         # Benchmark
        #         # linspacexmin = 0
        #         # linspacexmax = 10
        #         # linspaceymin = -1.5
        #         # linspaceymax = 3
        #
        #         # Levy
        #         # linspacexmin = -10
        #         # linspacexmax = 10
        #         # linspaceymin = -16
        #         # linspaceymax = 0
        #
        #         # Triangular wave
        #         # linspacexmin = 0
        #         # linspacexmax = 10
        #         # linspaceymin = -1.5
        #         # linspaceymax = 1.5
        #
        #         # Chirpwave wave
        #         # linspacexmin = 0
        #         # linspacexmax = 20
        #         # linspaceymin = -3
        #         # linspaceymax = 3
        #
        #         # Sinc Mixture
        #         # linspacexmin = -15
        #         # linspacexmax = 15
        #         # linspaceymin = -0.5
        #         # linspaceymax = 1.5
        #
        #         # # Gaussian Mixture
        #         # linspacexmin = 0
        #         # linspacexmax = 15
        #         # linspaceymin = -0.5
        #         # linspaceymax = 1.5
        #
        #         # Linear
        #         # linspacexmin = 0
        #         # linspacexmax = 2
        #         # linspaceymin = 0
        #         # linspaceymax = 0.5
        #
        #         # Linear Sin Function
        #         # linspacexmin = 0
        #         # linspacexmax = 10
        #         # linspaceymin = 0
        #         # linspaceymax = 10
        #
        #         # Gramacy Lee Function
        #         # linspacexmin = 0.5
        #         # linspacexmax = 2.5
        #         # linspaceymin = -1
        #         # linspaceymax = 1
        #
        #         # Ackley Function
        #         # linspacexmin = -10
        #         # linspacexmax = 10
        #         # linspaceymin = -25
        #         # linspaceymax = 0.5

        elif function_type == "OSC2D":
            linspacexmin = 0
            linspacexmax = 5
            linspaceymin = 0
            linspaceymax = 2

        elif function_type == "PAR2D":
            linspacexmin = -7
            linspacexmax = 7
            linspaceymin = -51
            linspaceymax = 1

        elif function_type == "LEVY2D":
            linspacexmin = -10
            linspacexmax = 10
            linspaceymin = -100
            linspaceymax = 1

        elif function_type == "ACKLEY2D":
            linspacexmin = -32.768
            linspacexmax = 32.768
            linspaceymin = -30
            linspaceymax = 0

        elif function_type == "BRANIN2D":
            linspacexmin = 0
            linspacexmax = 10
            linspaceymin = -1
            linspaceymax = 1

        elif function_type == "EGG2D":
            linspacexmin = -512
            linspacexmax = 512
            linspaceymin = -1000
            linspaceymax = 1000


        elif function_type == "HARTMANN3D":
            linspacexmin = -512
            linspacexmax = 512
            linspaceymin = -1000
            linspaceymax = 1000

        elif function_type == "HARTMANN6D":
            linspacexmin = -512
            linspacexmax = 512
            linspaceymin = -1000
            linspaceymax = 1000

        Xmin = linspacexmin
        Xmax = linspacexmax
        ymax = linspaceymax
        ymin = linspaceymin

        # For Ben1D and Lin1D Y- standardised
        ymax = 1
        ymin = 0

        if function_type == "OSC1D" or function_type == "BEN1D" or function_type == "GCL1D" or function_type == "ACK1D" or \
                function_type == "LIN1D" or function_type == "LINSIN1D" :
            number_of_dimensions = 1
            oned_bounds = [[linspacexmin, linspacexmax]]
            bounds = oned_bounds

        elif function_type == "BRANIN2D":
            number_of_dimensions = 2
            bounds = [[-5, 10], [0, 15]]
            Xmin = np.array([-5, 0])
            Xmax = np.array([10, 15])

        elif function_type == "OSC2D" or function_type == "PAR2D" or function_type == "ACKLEY2D" or \
                function_type == "EGG2D" or function_type == "LEVY2D":
            number_of_dimensions = 2
            bounds = [[linspacexmin, linspacexmax] for i in range(number_of_dimensions)]

        elif function_type == "HARTMANN3D":
            number_of_dimensions = 3
            bounds = [[linspacexmin, linspacexmax] for i in range(number_of_dimensions)]

        elif function_type == "HARTMANN6D":
            number_of_dimensions = 6
            bounds = [[linspacexmin, linspacexmax] for i in range(number_of_dimensions)]

        # sphere_bounds = [[linspacexmin, linspacexmax], [linspacexmin, linspacexmax]]
        # michalewicz2d_bounds = [[0, np.pi], [0, np.pi]]
        # random_bounds = [[0, 1], [1, 2]]
        # bounds = sphere_bounds
        # bounds = random_bounds

        lengthscale_bounds = [[0.1, 1] for i in range(number_of_dimensions)]
        signal_variance_bounds = [0.1, 1]
        fun_helper_obj = FunctionHelper(function_type)
        len_weights = [0.1, 0.3, 0.2, 0.1, 0.1, 0.2]
        len_weights_bounds = [[0.1, 1] for i in range(len(len_weights))]
        # controlled_obs = False
        controlled_obs = True

        if role != "ai" and role != "baseline":

            # Commenting for regression data - Forestfire
            random_points = []
            X = []

            # Generate specified (number of observed samples) random numbers for each dimension
            for dim in np.arange(number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1], number_of_random_observed_samples).reshape(1,
                                                                                                          number_of_random_observed_samples)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
            random_points = np.vstack(random_points)

            # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
            for sample_num in np.arange(number_of_random_observed_samples):
                array = []
                for dim_count in np.arange(number_of_dimensions):
                    array.append(random_points[dim_count, sample_num])
                X.append(array)
            X = np.vstack(X)

            if role == "GroundTruth":
                weight_params_estimation = True

                # Only for Oscillator function to force obs in the beginning of the space. Comment if any other true function
                if function_type == "OSC1D":
                    X = np.linspace(linspacexmin, 1.95, 15)
                    X = np.append(X, np.linspace(2, 4, 15))
                    X = np.append(X, np.linspace(4, linspacexmax, 5))
                    X = np.vstack(X)

            elif role == "HumanExpert":
                PH.printme(PH.p1, "Setting observation model for Human Expert ")
                weight_params_estimation = False

                if function_type == "OSC1D" and controlled_obs:
                    X = np.random.uniform(3.5, 7.5, number_of_random_observed_samples)
                    X = np.vstack(X)

                if function_type == "OSC2D" and controlled_obs:
                    random_points = []
                    X = []
                    bnds = [[2.5, 4.85], [linspacexmin, linspacexmax]]
                    for dim in np.arange(number_of_dimensions):
                        random_data_point_each_dim = np.random.uniform(bnds[dim][0], bnds[dim][1],number_of_random_observed_samples).reshape(1,
                                                                                                    number_of_random_observed_samples)
                        random_points.append(random_data_point_each_dim)

                    # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
                    random_points = np.vstack(random_points)

                    # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
                    for sample_num in np.arange(number_of_random_observed_samples):
                        array = []
                        for dim_count in np.arange(number_of_dimensions):
                            array.append(random_points[dim_count, sample_num])
                        X.append(array)
                    X = np.vstack(X)
                    # PH.printme(PH.p1, "Initial Observations X:", X)

                if function_type == "LIN1D" and controlled_obs:
                    X = np.random.uniform(0.5, 1.0, number_of_random_observed_samples)
                    # X = np.array([0.5, 0.75, 1.0])
                    X = np.vstack(X)

                elif function_type == "LINSIN1D" and controlled_obs:
                    # X = np.random.uniform(0, 1.5, number_of_random_observed_samples)
                    X = np.array([0.9, 1.0, 1.1])
                    X = np.vstack(X)

                elif function_type == "BEN1D" and controlled_obs:
                    X = np.random.uniform(7, 10, number_of_random_observed_samples)
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
                #
                # # objective function noisy
                # if function_type == "LIN1D":
                #     val = val + np.random.normal(0, np.sqrt(noise))
                y_arr.append(val)

            y_orig = np.vstack(y_arr)
            X = np.divide((X - Xmin), (Xmax - Xmin))

            # Standardising Y
            # y = (y - ymin) / (ymax - ymin)
            y = (y_orig - np.mean(y_orig))/np.std(y_orig)


            # ############### Uncomment to test the observation model # # # #
            # xs = np.linspace(linspacexmin, linspacexmax, 500)
            # ys = fun_helper_obj.get_true_func_value(xs)
            # xs = np.divide((xs - Xmin), (Xmax - Xmin))
            # ys = (ys - ymin) / (ymax - ymin)
            # import matplotlib.pyplot as plt
            # plt.plot(xs, ys)
            # plt.plot(X,y, "r+")
            # plt.show()
            ################################

        elif role == "ai":
            weight_params_estimation = False
            X = observations["observations_X"]
            y = observations["observations_y"]
            y_orig = observations["observations_y_orig"]

        elif role == "baseline":
            weight_params_estimation = True
            X = observations["observations_X"]
            y = observations["observations_y"]
            y_orig = observations["observations_y_orig"]

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

        ys_orig = np.vstack(ys_arr)
        # ys = sinc_function(Xs)

        Xs = np.divide((Xs - Xmin), (Xmax - Xmin))

        # Standardising y
        # ys = (ys - ymin) / (ymax - ymin)
        ys = (ys_orig - np.mean(y_orig)) / (np.std(y_orig))

        gp_object = GaussianProcessRegressor(start_time, role, kernel_type, number_of_test_datapoints, noise, linspacexmin, linspacexmax,
                                             linspaceymin, linspaceymax, signal_variance, number_of_dimensions,
                                             number_of_random_observed_samples, X, y, y_orig, number_of_restarts_likelihood, bounds,
                                             lengthscale_bounds,
                                             signal_variance_bounds, Xmin, Xmax, ymin, ymax, Xs, ys, ys_orig, char_len_scale, len_weights,
                                             len_weights_bounds, weight_params_estimation, fun_helper_obj)
        return gp_object


