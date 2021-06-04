from GP_Regression import GaussianProcess
from Functions import FunctionHelper
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH


class GaussianProcessWrapper:
    def __init__(self, synthetic_data):
        self.gaussian_object = None
        self.posterior_plot = None
        self.synthetic_data = synthetic_data

    def construct_gp_regressor_real(self, stamp, dataset, posterior_plot):

        kernel_type = 'SE'
        char_len_scale = 0.3
        number_of_test_datapoints = 500
        noise = 0.0
        random_seed = 500
        signal_variance = 1
        degree = 2
        self.posterior_plot = posterior_plot

        bounds = [[0, np.pi], [0, np.pi]]
        if dataset == 'challenger':
            x_cols = [3, 4]
            y_cols = [2]
            number_of_dimensions = 2
            number_of_observed_samples = 23
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0, 100], [0, 250]]

            Xmin = np.array([0, 0])
            Xmax = np.array([100, 250])
            ymin = 0
            ymax = 2

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            D = np.array([
                [66, 50],
                [70, 50],
                [69, 50],
                [68, 50],
                [67, 50],
                [72, 50],
                [73, 100],
                [70, 100],
                [57, 200],
                [63, 200],
                [70, 200],
                [78, 200],
                [67, 200],
                [53, 200],
                [67, 200],
                [75, 200],
                [70, 200],
                [81, 200],
                [76, 200],
                [79, 200],
                [75, 200],
                [76, 200],
                [58, 200]])

            f = np.array([
                [0],
                [1],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [1],
                [1],
                [1],
                [0],
                [0],
                [2],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [1]])

        elif dataset == 'concrete':

            number_of_dimensions = 8
            number_of_observed_samples = 1029
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0,1] for i in range(0,8)]

            Xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            Xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            ymin = 0
            ymax = 1

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            file = '../../DatasetUtils/Dataset/Concrete_Data.csv'
            total_data = pd.read_csv(file)

            min_max_scaler = MinMaxScaler()
            total_data_std = min_max_scaler.fit_transform(total_data)

            D = total_data_std[:, 0:-1]
            f = total_data_std[:, -1]

        elif dataset == 'fertility':

            number_of_dimensions = 9
            number_of_observed_samples = 100
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0, 1] for i in range(0, 9)]

            Xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            Xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
            ymin = 0
            ymax = 1

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            file = '../../DatasetUtils/Dataset/Fertility_Diagnosis.csv'
            total_data = pd.read_csv(file)

            min_max_scaler = MinMaxScaler()
            total_data_std = min_max_scaler.fit_transform(total_data)

            D = total_data_std[:, 0:-1]
            f = total_data_std[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(D, f, test_size=0.20)



        X = X_train
        Xs = X_test
        y = y_train.reshape(-1, 1)
        ys = y_test.reshape(-1, 1)

        lengthscale_bounds = [[0.1, 10]]
        signal_variance_bounds = [0.1, 10]
        true_func_type = "custom"
        func_helper_obj = FunctionHelper(true_func_type)
        len_weights = [0.1, 0.3, 0.2]
        len_weights_bounds = [[0.1, 5] for i in range(3)]
        weight_params_estimation = False
        degree_estimation = False

        PH.printme(PH.p1, X.shape)
        gaussian_object = GaussianProcess(stamp, kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                          linspacexmax, linspaceymin, linspaceymax, signal_variance,
                                          number_of_dimensions, number_of_observed_samples, X, y, hyper_params_estimation,
                                          number_of_restarts_likelihood, lengthscale_bounds, signal_variance_bounds, func_helper_obj,
                                          Xmin, Xmax, ymin, ymax, Xs, ys, char_len_scale, len_weights, len_weights_bounds,
                                          weight_params_estimation, degree_estimation, degree)

        self.gaussian_object = gaussian_object
        return

    def construct_gp_regressor_syn(self, stamp, posterior_plot):

        kernel_type = 'SE'
        char_len_scale = 0.3
        number_of_test_datapoints = 500
        noise = 0.0
        random_seed = 500
        signal_variance = 1
        degree = 2
        self.posterior_plot = posterior_plot


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
        # linspacexmin = -1
        # linspacexmax = 10
        # linspaceymin = -0.5
        # linspaceymax = 1.5

        # Triangular wave
        linspacexmin = 0
        linspacexmax = 10
        linspaceymin = -1.5
        linspaceymax = 1.5

        # Chirp wave
        # linspacexmin = 0
        # linspacexmax = 20
        # linspaceymin = -3
        # linspaceymax = 3

        # # Sinc Mixture
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


        number_of_dimensions = 1
        number_of_observed_samples = 40
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
        degree_estimation = False
        posterior_plot = False
        # posterior_plot = True

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
        # x_obs = np.linspace(-15, -5, 20)
        # x_obs = np.append(x_obs, np.linspace(5, 15, 20))
        # X = x_obs.reshape(-1, 1)

        # SINC Replicating same observations as that of SE, MAT, MKL
        # X = np.array([[ -8.38751903],
        #               [-5.79460269],
        #               [-5.44723374],
        #               [-6.76226188],
        #               [-13.65178808],
        #               [-7.29795703],
        #               [-9.26582284],
        #               [-7.54705499],
        #               [-12.34416592],
        #               [-6.1399037],
        #               [-12.19930636],
        #               [-12.47204447],
        #               [-10.62280772],
        #               [-7.73649382],
        #               [-13.87740007],
        #               [-14.44905393],
        #               [-5.96014355],
        #               [-8.65370561],
        #               [-5.65379244],
        #               [-5.91076591],
        #               [6.03145514],
        #               [12.39865955],
        #               [12.5229122],
        #               [8.43562893],
        #               [14.89925025],
        #               [10.91264131],
        #               [7.08724195],
        #               [12.82172385],
        #               [13.15508181],
        #               [11.89432534],
        #               [11.71068664],
        #               [12.99645536],
        #               [14.7930659],
        #               [8.43028352],
        #               [14.93659035],
        #               [8.50961315],
        #               [14.93440112],
        #               [10.78275833],
        #               [12.37612112],
        #               [12.2090651]])

        # Gaussian Mixtures
        # x_obs = np.linspace(0, 5, 20)
        # # x_obs = np.append(x_obs, np.array([6.6,6.5,6.3,6.0,5.6,8.4,8.5,8.6,9.0,9.5]))
        # x_obs = np.append(x_obs, np.linspace(10, 15, 20))
        # X = x_obs.reshape(-1, 1)

        # GMIX Replicating same observations as that of SE, MAT, MKL
        # X = np.array([[3.30624048],
        #               [4.60269866],
        #               [4.77638313],
        #               [4.11886906],
        #               [0.67410596],
        #               [3.85102148],
        #               [2.86708858],
        #               [3.72647251],
        #               [1.32791704],
        #               [4.43004815],
        #               [1.40034682],
        #               [1.26397776],
        #               [2.18859614],
        #               [3.63175309],
        #               [0.56129997],
        #               [0.27547303],
        #               [4.51992823],
        #               [3.1731472],
        #               [4.67310378],
        #               [4.54461704],
        #               [10.51572757],
        #               [13.69932977],
        #               [13.7614561],
        #               [11.71781447],
        #               [14.94962513],
        #               [12.95632065],
        #               [11.04362097],
        #               [13.91086192],
        #               [14.0775409],
        #               [13.44716267],
        #               [13.35534332],
        #               [13.99822768],
        #               [14.89653295],
        #               [11.71514176],
        #               [14.96829517],
        #               [11.75480658],
        #               [14.96720056],
        #               [12.89137917],
        #               [13.68806056],
        #               [13.60453255]])

        # Triangular  wave - 2 peaks # Replicating same observations as that of SE, MKL, MAT
        X = np.array([[1.98374429],
                      [2.76161919],
                      [2.86582988],
                      [2.47132143],
                      [0.40446358],
                      [2.31061289],
                      [1.72025315],
                      [2.2358835],
                      [0.79675022],
                      [2.65802889],
                      [0.84020809],
                      [0.75838666],
                      [1.31315769],
                      [2.17905185],
                      [0.33677998],
                      [0.16528382],
                      [2.71195694],
                      [1.90388832],
                      [2.80386227],
                      [2.72677023],
                      [7.81258206],
                      [8.95946382],
                      [9.00916488],
                      [7.37425157],
                      [9.9597001],
                      [8.36505652],
                      [8.03489678],
                      [9.12868954],
                      [9.26203272],
                      [8.75773013],
                      [8.68427466],
                      [9.19858215],
                      [9.91722636],
                      [7.37211341],
                      [9.97463614],
                      [7.40384526],
                      [9.97376045],
                      [8.31310333],
                      [5.40044845],
                      [3.53362604]])

        # # Linear
        # x_obs = np.linspace(linspacexmin, 0.5, 5)
        # x_obs = np.append(x_obs, np.linspace(1.5, linspacexmax, 5))
        # X = x_obs.reshape(-1, 1)

        # Linear Sin Function
        # x_obs = np.linspace(linspacexmin, 3, 15)
        # x_obs = np.append(x_obs, np.linspace(7, linspacexmax, 15))
        # X = x_obs.reshape(-1, 1)


        PH.printme(PH.p1, "Data points: ",X)
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

        ys = None
        if self.posterior_plot:
            ys_arr = []
            for each_xs in Xs:
                val_xs = func_helper_obj.get_true_func_value(each_xs)
                ys_arr.append(val_xs)
            ys = np.vstack(ys_arr)
            # ys = sinc_function(Xs)
            ys = (ys - ymin) / (ymax - ymin)

        Xs = np.divide((Xs - Xmin), (Xmax - Xmin))

        gaussian_object = GaussianProcess(stamp, kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                         linspacexmax, linspaceymin, linspaceymax, signal_variance,
                                         number_of_dimensions, number_of_observed_samples, X, y, hyper_params_estimation,
                                         number_of_restarts_likelihood, lengthscale_bounds, signal_variance_bounds, func_helper_obj,
                                         Xmin, Xmax, ymin, ymax, Xs, ys, char_len_scale, len_weights, len_weights_bounds,
                                         weight_params_estimation, degree_estimation, degree)

        self.gaussian_object = gaussian_object
        return

    def compute_likelihood_for_kernel(self, kernel_type, observations_kernel, optional_hypergp_obj):
        count = 1

        if self.synthetic_data:
            likelihood = self.gaussian_object.runGaussian(count, kernel_type, observations_kernel, optional_hypergp_obj)
        else:
            likelihood = self.gaussian_object.compute_predictive_likelihood(count, kernel_type, observations_kernel, optional_hypergp_obj)
        return likelihood

    def compute_posterior_distribution(self, kernel_type, observations_kernel, optional_hypergp_obj, msg):
        self.gaussian_object.plot_posterior_distribution(kernel_type, observations_kernel, optional_hypergp_obj, msg)

    def plot_GP_Reg__kernel(self, msg, observations_kernel):
        self.gaussian_object.kernel_type = "HYPER"
        self.gaussian_object.plot_kernel(msg, observations_kernel)
