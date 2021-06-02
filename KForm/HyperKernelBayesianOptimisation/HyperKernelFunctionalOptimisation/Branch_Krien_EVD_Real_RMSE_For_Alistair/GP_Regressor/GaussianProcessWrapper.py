from GP_Regression import GaussianProcess
from Functions import FunctionHelper
import numpy as np
import pandas as pd
from sklearn import preprocessing
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

            file = 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\DatasetUtils' \
                   '\Dataset\Concrete_Data.csv'
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

            file = 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\DatasetUtils' \
                   '\Dataset\Fertility_Diagnosis.csv'
            total_data = pd.read_csv(file)

            min_max_scaler = MinMaxScaler()
            total_data_std = min_max_scaler.fit_transform(total_data)

            D = total_data_std[:, 0:-1]
            f = total_data_std[:, -1]

        elif dataset == 'yacht':

            number_of_dimensions = 6
            number_of_observed_samples = 308
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0, 1] for i in range(0, 6)]

            Xmin = np.array([0, 0, 0, 0, 0, 0])
            Xmax = np.array([1, 1, 1, 1, 1, 1])
            ymin = 0
            ymax = 1

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            file = 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\DatasetUtils' \
                   '\Dataset\yacht2.csv'
            total_data = pd.read_csv(file)

            min_max_scaler = MinMaxScaler()
            total_data_std = min_max_scaler.fit_transform(total_data)

            # D = total_data.iloc[:, 0:-1].to_numpy()
            # f = total_data.iloc[:, -1].to_numpy()

            D = total_data_std[:, 0:-1]
            f = total_data_std[:, -1]

            print(D, "\n\n",f)

        elif dataset == 'boston':

            number_of_dimensions = 13
            number_of_observed_samples = 506
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0, 1] for i in range(0, 13)]

            Xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            Xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            ymin = 0
            ymax = 1

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            file = 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\DatasetUtils' \
                   '\Dataset\housing_boston.csv'
            total_data = pd.read_csv(file)

            min_max_scaler = MinMaxScaler()
            total_data_std = min_max_scaler.fit_transform(total_data)

            D = total_data_std[:, 0:-1]
            f = total_data_std[:, -1]


        elif dataset == 'airfoil':

            number_of_dimensions = 5
            number_of_observed_samples = 1503
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0, 1] for i in range(0, 13)]

            Xmin = np.array([0, 0, 0, 0, 0])
            Xmax = np.array([1, 1, 1, 1, 1])
            ymin = 0
            ymax = 1

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            file = 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\DatasetUtils' \
                   '\Dataset\dataset_airfoil.csv'
            total_data = pd.read_csv(file)

            min_max_scaler = MinMaxScaler()
            total_data_std = min_max_scaler.fit_transform(total_data)

            D = total_data_std[:, 0:-1]
            f = total_data_std[:, -1]


        elif dataset == 'concreteslump':

            number_of_dimensions = 9
            number_of_observed_samples = 103
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0, 1] for i in range(0, number_of_dimensions)]

            Xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            Xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
            ymin = 0
            ymax = 1

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            file = 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\DatasetUtils' \
                   '\Dataset\concreteslump.csv'
            total_data = pd.read_csv(file)

            min_max_scaler = MinMaxScaler()
            total_data_std = min_max_scaler.fit_transform(total_data)

            D = total_data_std[:, 1:-1]
            f = total_data_std[:, -1]
            print("concrete slump test")


        elif dataset == 'auto':

            number_of_dimensions = 9
            number_of_observed_samples = 103
            hyper_params_estimation = True
            number_of_restarts_likelihood = 100
            bounds = [[0, 1] for i in range(0, number_of_dimensions)]

            Xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            Xmax = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
            ymin = 0
            ymax = 1

            linspacexmin = 0
            linspacexmax = 1
            linspaceymin = 0
            linspaceymax = 1

            file = 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\DatasetUtils' \
                   '\Dataset\Auto.csv'
            total_dataframe = pd.read_csv(file,  na_values='?')

            dataframe = total_dataframe.fillna(method='bfill')

            f = dataframe.iloc[:, 0]
            dataframe = dataframe.drop(dataframe.columns[0], axis=1)
            D = dataframe

            dataframe.astype({'col3': 'float64'}).dtypes

            dataframe_num = dataframe.select_dtypes(include=[np.number])
            min_max_scaler = MinMaxScaler()
            dataframe_num_std = min_max_scaler.fit_transform(dataframe_num)

            dataframe_cat = dataframe.select_dtypes(include=[object])

            label_encoder = preprocessing.LabelEncoder()
            labeled_dataframe_cat = dataframe_cat.apply(label_encoder.fit_transform)
            #
            onehot_encoder = preprocessing.OneHotEncoder()
            onehot_encoder.fit(labeled_dataframe_cat)
            onehotlabel_cols = onehot_encoder.transform(labeled_dataframe_cat).toarray()

            D_std = np.concatenate((dataframe_num_std, onehotlabel_cols), axis=1)

            D = D_std
            min_max_scaler = MinMaxScaler()
            f = f.to_numpy()
            f = f.reshape(-1, 1)
            f = min_max_scaler.fit_transform(f)

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

        # # Normalising code commented
        # X = np.divide((X - Xmin), (Xmax - Xmin))
        # y = (y - ymin) / (ymax - ymin)
        #
        # random_points = []
        # Xs = []
        #
        # # Generate specified (number of unseen data points) random numbers for each dimension
        # for dim in np.arange(number_of_dimensions):
        #     random_data_point_each_dim = np.linspace(bounds[dim][0], bounds[dim][1],
        #                                              number_of_test_datapoints).reshape(1, number_of_test_datapoints)
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
        # # Obtain the values for the true function, so that true function can be plotted in the case of 1D currently
        #
        # # Comented to accomodate Oscillator function mergen input domain
        # # ys = func_helper_obj.get_true_func_value(Xs)
        # ys = None
        # if self.posterior_plot:
        #     ys_arr = []
        #     for each_xs in Xs:
        #         val_xs = func_helper_obj.get_true_func_value(each_xs)
        #         ys_arr.append(val_xs)
        #     ys = np.vstack(ys_arr)
        #     # ys = sinc_function(Xs)
        #     ys = (ys - ymin) / (ymax - ymin)
        #
        # # Normalising code commented
        # Xs = np.divide((Xs - Xmin), (Xmax - Xmin))


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
        # linspacexmin = 0
        # linspacexmax = 4
        # linspaceymin = -1.5
        # linspaceymax = 1.5

        # Triangular wave
        # linspacexmin = 0
        # linspacexmax = 10
        # linspaceymin = -1.5
        # linspaceymax = 1.5

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
        linspacexmin = 0
        linspacexmax = 10
        linspaceymin = 0
        linspaceymax = 10


        number_of_dimensions = 1
        number_of_observed_samples = 30
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

        # Gaussian Mixtures
        # x_obs = np.linspace(0, 5, 20)
        # x_obs = np.append(x_obs, np.linspace(10, 15, 20))
        # X = x_obs.reshape(-1, 1)

        # # Linear
        # x_obs = np.linspace(linspacexmin, 0.5, 5)
        # x_obs = np.append(x_obs, np.linspace(1.5, linspacexmax, 5))
        # X = x_obs.reshape(-1, 1)

        # Linear Sin Function
        x_obs = np.linspace(linspacexmin, 3, 15)
        x_obs = np.append(x_obs, np.linspace(7, linspacexmax, 15))
        X = x_obs.reshape(-1, 1)

        # Selecting observations for Chirpwave
        # X = np.array([
        #     [0.320835626],
        #     [1.40835626],
        #     [2.45835626],
        #     [3.20835626],
        #     [4.20835626],
        #     [5.60195798],
        #     [6.42273538],
        #     [7.21350942],
        #     [7.32440451],
        #     [7.90691315],
        #     [8.32949304],
        #     [8.67509187],
        #     [8.90218965],
        #     [9.55831041],
        #     [10.43239387],
        #     [10.64328088],
        #     [10.94865737],
        #     [11.6755274],
        #     [12.354817602],
        #     [12.93994599],
        #     [13.20932905],
        #     [14.22347481],
        #     [14.3232872],
        #     [14.5357832],
        #     [15.4473578],
        #     [15.973578],
        #     [17.34861657],
        #     [18.16232872],
        #     [18.95330988],
        #     [19.95330988]])

        # Held out observations
        # X = np.array([[2.19214345],
        #  [2.43323017],
        #  [2.96276434],
        #  [3.87453034],
        #  [3.91497631],
        #  [4.03951311],
        #  [4.54332442],
        #  [6.77077269],
        #  [7.11277749],
        #  [7.59546128],
        #  [7.73820903],
        #  [7.97831401],
        #  [15.36052736],
        #  [15.74902047],
        #  [16.08084353],
        #  [16.31432616],
        #  [16.35883791],
        #  [17.64481183],
        #  [17.86025608],
        #  [19.99155822]])

        # Modified Held out with 70 Obs
 #        X= np.array([[ 0.        ],
 # [ 0.19230769],
 # [ 0.38461538],
 # [ 0.57692308],
 # [ 0.76923077],
 # [ 0.96153846],
 # [ 1.15384615],
 # [ 1.34615385],
 # [ 1.53846154],
 # [ 1.73076923],
 # [ 1.92307692],
 # [ 2.11538462],
 # [ 2.30769231],
 # [ 2.5       ],
 # [ 2.69230769],
 # [ 2.88461538],
 # [ 3.07692308],
 # [ 3.26923077],
 # [ 3.46153846],
 # [ 3.65384615],
 # [ 3.84615385],
 # [ 4.03846154],
 # [ 4.23076923],
 # [ 4.42307692],
 # [ 4.61538462],
 # [ 4.80769231],
 # [ 5.        ],
 # [ 5.19230769],
 # [ 5.38461538],
 # [ 5.57692308],
 # [ 5.76923077],
 # [ 5.96153846],
 # [ 6.15384615],
 # [ 6.34615385],
 # [ 6.53846154],
 # [ 6.73076923],
 # [ 6.92307692],
 # [ 7.11538462],
 # [ 7.30769231],
 # [ 7.5       ],
 # [8.2        ],
 # [8.913793103],
 # [9.827586207],
 # [10.64137931 ],
 # [16.55172414],
 # [16.68965517],
 # [16.82758621],
 # [16.96551724],
 # [17.10344828],
 # [17.24137931],
 # [17.37931034],
 # [17.51724138],
 # [17.65517241],
 # [17.79310345],
 # [17.93103448],
 # [18.06896552],
 # [18.20689655],
 # [18.34482759],
 # [18.48275862],
 # [18.62068966],
 # [18.75862069],
 # [18.89655172],
 # [19.03448276],
 # [19.17241379],
 # [19.31034483],
 # [19.44827586],
 # [19.5862069 ],
 # [19.72413793],
 # [19.86206897],
 # [20.        ]])

        # Modified Held out with 50 Obs
        # X = np.array([[0.],
        #               [0.19230769],
        #               [0.38461538],
        #               [0.57692308],
        #               [0.76923077],
        #               [0.96153846],
        #               [1.15384615],
        #               [1.34615385],
        #               [1.53846154],
        #               [1.73076923],
        #               [1.92307692],
        #               [2.11538462],
        #               [2.30769231],
        #               [2.5],
        #               [2.69230769],
        #               [2.88461538],
        #               [3.07692308],
        #               [3.26923077],
        #               [3.46153846],
        #               [3.65384615],
        #               [3.84615385],
        #               [4.03846154],
        #               # [4.23076923],
        #               [4.42307692],
        #               # [4.61538462],
        #               [4.80769231],
        #               # [5.],
        #               [5.19230769],
        #               # [5.38461538],
        #               [5.57692308],
        #               # [5.76923077],
        #               [5.96153846],
        #               # [6.15384615],
        #               [6.34615385],
        #               # [6.53846154],
        #               [6.73076923],
        #               [6.92307692],
        #               [7.11538462],
        #               [7.30769231],
        #               [7.5],
        #               [8.2],
        #               [8.5],
        #               [8.913793103],
        #               [9.4],
        #               [9.827586207],
        #               [16.4137931],
        #               # [16.55172414],
        #               [16.68965517],
        #               # [16.82758621],
        #               [16.96551724],
        #               # [17.10344828],
        #               [17.24137931],
        #               # [17.37931034],
        #               [17.51724138],
        #               # [17.65517241],
        #               [17.79310345],
        #               # [17.93103448],
        #               [18.06896552],
        #               # [18.20689655],
        #               [18.34482759],
        #               # [18.48275862],
        #               [18.62068966],
        #               # [18.75862069],
        #               [18.89655172],
        #               # [19.03448276],
        #               [19.17241379],
        #               # [19.31034483],
        #               [19.44827586],
        #               # [19.5862069],
        #               [19.72413793],
        #               # [19.86206897],
        #               [20.]])

        PH.printme(PH.p1, X)
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

        ys = None
        if self.posterior_plot:
            ys_arr = []
            for each_xs in Xs:
                val_xs = func_helper_obj.get_true_func_value(each_xs)
                ys_arr.append(val_xs)
            ys = np.vstack(ys_arr)
            # ys = sinc_function(Xs)
            ys = (ys - ymin) / (ymax - ymin)

        #Normalising code commented
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

    def compute_rmse_for_kernel(self, kernel_type, observations_kernel, optional_hypergp_obj):
        count = 1
        if self.synthetic_data:
            likelihood = self.gaussian_object.runGaussian(count, kernel_type, observations_kernel, optional_hypergp_obj)
        else:
            rmse = self.gaussian_object.compute_rmse(count, kernel_type, observations_kernel, optional_hypergp_obj)
        return rmse

    def compute_posterior_distribution(self, kernel_type, observations_kernel, optional_hypergp_obj, msg):
        self.gaussian_object.plot_posterior_distribution(kernel_type, observations_kernel, optional_hypergp_obj, msg)

    def plot_GP_Reg__kernel(self, msg, observations_kernel):
        self.gaussian_object.kernel_type = "HYPER"
        self.gaussian_object.plot_kernel(msg, observations_kernel)
