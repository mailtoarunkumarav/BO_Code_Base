from ScratchGPModular import GaussianProcess
from AcquisitionFunction import AcquisitionFunction
from BayesianOptimization import BayesianOptimization
from Functions import FunctionHelper, Custom_Print
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys


# setting up the global parameters for plotting graphs i.e, graph size and suppress warning from multiple graphs
# being plotted
plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams["font.size"] = 12
plt.rcParams['figure.max_open_warning'] = 0
# np.seterr(divide='ignore', invalid='ignore')

# To fix the random number genration, currently not able, so as to retain the random selection of points
random_seed = 200
np.random.seed(random_seed)

# Class for starting Bayesian Optimization with the specified parameters
class BayesianOptimizationWrapper:

    def opt_wrapper(start_time):
        # Required for plotting graphs in the casse of 1D

        # #GCLEE
        linspacexmin = 0.5;linspacexmax = 2.5;linspaceymin = -6;linspaceymax = 1

        # #Levy Function
        # linspacexmin = -10;linspacexmax = 10;linspaceymin = -16;linspaceymax = 0

        # Epsilons is the value used during the maximization of PI and EI ACQ functions
        # Greater value like Epsilon = 10 is more of exploration and Epsilon = 0.0001 (exploitation)
        # Epsilon1 used in PI : 3
        epsilon1 = 3
        # epsilon2 used in EI : 4
        epsilon2 = 4

        # Initial value to denote the type of ACQ function to be used, but ignored as all ACQs are run in the sequence
        # acq_fun_list = ['ei', 'pi', 'rs', 'ucb']
        acq_fun_list = ['ei']

        # Number of points required to be observed to evaluate the unknown function ((10:20)*no_dimensions )
        # number_of_iterations = number_of_dimensions * 10
        number_of_iterations = 40

        # List of true functions to choose from
        # 1D functions : sin, cos, custom
        # 2D functions : branin2d, sphere
        # true_func = 'sin'
        # true_func = 'cos'
        # true_func = 'custom'
        # true_func = 'branin2d'
        # true_func = 'hartmann3d'
        # true_func = 'hartmann6d'
        # true_func = 'sphere'
        # true_func = 'syn2d'
        # true_func = 'levy2d'
        true_func = 'ackley2d'
        # true_func = 'egg2d'
        # true_func = 'michalewicz2d'

        if true_func == 'sin' or true_func == 'cos' or true_func == 'custom':

            # Number of the dimensions being used in the optimization
            number_of_dimensions = 1
            # Different bounds used for different problems
            # bounds used for 1D Sin and custom function
            oned_bounds = [[linspacexmin, linspacexmax]]
            bounds = oned_bounds
            epsilon1 = 0.005
            epsilon2 = 0.001
            # epsilon2 = 0.01 #Exploration
            number_of_iterations = 15

            # Bounds for length scale parameters
            len_scale_func_type = ['linear']
            len_scale_params = np.array([[0.06, 0.01]])
            len_scale_param_bounds = [[[0.001, 2],[0.001, 2]]]
            # len_scale_func_type = ['gaussian']
            # len_scale_params = np.array([[0.5, 0.2]])
            # len_scale_param_bounds = [[[0.3, 0.6], [0.2, 0.3]]]

            # #Bounds for Normalization
            # #Levy
            # Xmin = np.array([-10]);Xmax = np.array([10]);ymin = -16;ymax = 0

            # #GCLee
            Xmin = np.array([0.5]);Xmax = np.array([2.5]);ymin = -6;ymax = 1

        elif true_func == 'branin2d':
            number_of_dimensions = 2
            branin_bounds = [[-5, 10], [0, 15]]
            bounds = branin_bounds
            epsilon1 = 0.001
            epsilon2 = 0.001
            number_of_iterations = 20

            len_scale_func_type = ['linear', 'linear']
            len_scale_params = np.array([[0.3, 0.1], [0.3, 0.1]])
            len_scale_param_bounds = [[[0.001, 1], [0.01, 1]], [[0.01, 1], [0.01, 1]]]

            # #Bounds for Normalization
            Xmin = np.array([0])
            Xmax = np.array([10])
            ymin = -1
            ymax = 1

        elif true_func == 'sphere':
            number_of_dimensions = 2
            sphere_bounds = [[0, 10], [0, 10]]
            bounds = sphere_bounds

        elif true_func == 'hartmann3d':
            number_of_dimensions = 3
            hartmann3d_bounds = [[0, 1] for nd in np.arange(number_of_dimensions)]
            bounds = hartmann3d_bounds
            epsilon1 = 0.01
            epsilon2 = 0.01
            number_of_iterations = 20

            len_scale_params = np.array([[0.06, 0.01], [0.06, 0.01], [0.06, 0.01]])
            len_scale_func_type = ['linear', 'linear', 'linear']
            len_scale_param_bounds = [[[0.01, 1],[0.01, 1]], [[0.01, 1],[0.01, 1]], [[0.01, 1],[0.01, 1]]]

        elif (true_func == 'hartmann6d'):
            number_of_dimensions = 6
            hartmann6d_bounds = [[0, 1] for nd in np.arange(number_of_dimensions)]
            bounds = hartmann6d_bounds
            epsilon1 = 0.1
            epsilon2 = 0.1
            number_of_iterations = 20

        elif true_func == 'syn2d':
            number_of_dimensions = 2
            syn2d_bounds = [[0.5, 2.5], [0.5, 2.5]]
            bounds = syn2d_bounds
            epsilon1 = 0.001
            epsilon2 = 0.001
            number_of_iterations = 50
            len_scale_func_type = ['linear', 'linear']
            len_scale_params = np.array([[0.3, 0.1], [0.3, 0.1]])
            len_scale_param_bounds = [[[0.001, 1], [0.001, 1]], [[0.001, 1], [0.001, 1]]]

            # #Bounds for Normalization
            Xmin = np.array([0.5])
            Xmax = np.array([2.5])
            ymin = -3
            ymax = 3

        elif true_func == 'levy2d':
            number_of_dimensions = 2
            levy2d_bounds = [[-10,10], [-10,10]]
            bounds = levy2d_bounds
            epsilon1 = 0.001
            epsilon2 = 0.001
            number_of_iterations = 30
            len_scale_func_type = ['gaussian', 'gaussian']
            len_scale_params = np.array([[0.5, 0.2], [0.5, 0.2]])
            len_scale_param_bounds = [[[0.3, 0.6], [0.2, 0.3]], [[0.3, 0.6], [0.2, 0.3]]]

            # #Bounds for Normalization
            Xmin = np.array([-10])
            Xmax = np.array([10])
            ymin = -100
            ymax = 1

        elif true_func == 'ackley2d':
            number_of_dimensions = 2
            ackley2d_bounds = [[-32.768,32.768], [-32.768,32.768]]
            bounds = ackley2d_bounds
            epsilon1 = 0.001
            epsilon2 = 0.01
            number_of_iterations = 30
            # len_scale_func_type = ['gaussian', 'gaussian']
            # len_scale_params = np.array([[0.5, 0.2], [0.5, 0.2]])
            # len_scale_param_bounds = [[[0.3, 0.6], [0.2, 0.3]], [[0.3, 0.6], [0.2, 0.3]]]

            len_scale_func_type = ['quadratic', 'quadratic']
            len_scale_params = np.array([[-2, 1, 1.1], [-2, 1, 1.1]])
            len_scale_param_bounds = [[[2, 2.1], [-2.1, 2], [0.65,0.75]], [[2, 2.1], [-2.1, 2], [0.65,0.75]]]

            # len_scale_func_type = ['linear', 'linear']
            # len_scale_params = np.array([[0.3, 0.1], [0.3, 0.1]])
            # len_scale_param_bounds = [[[0.001, 1], [0.001, 1]], [[0.001, 1], [0.001, 1]]]

            # #Bounds for Normalization
            Xmin = np.array([-32.768])
            Xmax = np.array([32.768])
            ymin = -30
            ymax = 0


        elif true_func == 'egg2d':
            number_of_dimensions = 2
            ackley2d_bounds = [[-512,512], [-512,512]]
            bounds = ackley2d_bounds
            epsilon1 = 0.001
            epsilon2 = 0.001
            number_of_iterations = 30
            len_scale_func_type = ['gaussian', 'gaussian']
            len_scale_params = np.array([[0.5, 0.2], [0.5, 0.3]])
            len_scale_param_bounds = [[[0.5, 0.6], [0.6, 0.8]], [[0.5, 0.6], [0.6, 0.8]]]

            # len_scale_func_type = ['quadratic', 'quadratic']
            # # len_scale_params = np.array([[3.5, 3, 0.55], [3.5, 3, 0.55]])
            # len_scale_params = np.array([[2, 1, 0.2], [2, 1, 0.2]])
            # len_scale_param_bounds = [[[2, 4], [0.2, 0.3]], [[0.3, 0.6], [0.2, 0.3]]]

            # #Bounds for Normalization
            Xmin = np.array([-512])
            Xmax = np.array([512])
            ymin = -1000
            ymax = 1000

        elif true_func == 'michalewicz2d':
            number_of_dimensions = 2
            michalewicz2d_bounds = [[0, np.pi], [0, np.pi]]
            bounds = michalewicz2d_bounds
            epsilon1 = 0.001
            epsilon2 = 0.001
            number_of_iterations = 30
            len_scale_func_type = ['gaussian', 'gaussian']
            len_scale_params = np.array([[0.5, 0.2], [0.5, 0.3]])
            len_scale_param_bounds = [[[0.5, 0.55], [0.6, 0.8]], [[0.5, 0.55], [0.6, 0.8]]]

            # len_scale_func_type = ['quadratic', 'quadratic']
            # # len_scale_params = np.array([[3.5, 3, 0.55], [3.5, 3, 0.55]])
            # len_scale_params = np.array([[2, 1, 0.2], [2, 1, 0.2]])
            # len_scale_param_bounds = [[[2, 4], [0.2, 0.3]], [[0.3, 0.6], [0.2, 0.3]]]

            # #Bounds for Normalization
            Xmin = np.array([0])
            Xmax = np.array([np.pi])
            ymin = 0
            ymax = 2


        # Number of samples for which we have the y values to be known (no_dimensions+1)
        number_of_observed_samples = number_of_dimensions + 1

        # Number of restarts required during the calculation of the maxima during the maximization of acq functions
        number_of_restarts_acq = 100

        # Number of restarts required during the calculation of the maxima during the maximization of likelihood
        number_of_restarts_likelihood = 100

        # Number of runs the BO has to be run for calculating the regret and the corresponding means and the standard devs
        number_of_runs = 3

        # Type of kernel to be used in the optimization process
        # 0 - Squared Exponential; 1 - Rational Quadratic Function; 2 - Exponential; 3 - Periodic***(To be fixed)
        kernel_type = 0

        # Characteristics of the Kernel being used, currently implemented for the square exponential kernel
        # Required to set up the GP Object initially
        kernel_char = 'ard'

        # Characteristic length scale to be used in the kernel function
        len_scale_bounds = [[0.3, 3] for nd in np.arange(number_of_dimensions)]

        # characteristic_length_scale = np.array([1 for nd in np.arange(number_of_dimensions)])
        char_length_scale = [0.5 for nd in np.arange(number_of_dimensions)]

        # Boolean to specify estimation of length scale Parameters
        params_estimation = False

        # Boolean to specify estimation of length scale itself : False signifies fixed length scale overall
        len_scale_estimation = False

        # Signal variance bounds
        signal_variance_bounds = [0.1, 1]

        multi_len_scales = [0.7,0.4,0.3,0.1]
        len_weights = [1,1,1,1]
        len_weight_bounds= [[0.1,1],[0.1,1],[0.1,1],[0.1,1]]
        weights_estimation = False

        # Initial Signal Variance
        signal_variance = 1

        # Number of unseen data points in order to evaluate the function
        number_of_test_datapoints = 100

        # Noise to be added in the system***
        noise = 0.0

        # Kappa value to be used during the maximization of UCB ACQ function, but this is overriden in the case if
        # Kappa is calculated at each iteration as a function of the iteration and other parameters
        # kappa=10 is more of exploration and kappa = 0.1 is more of exploitation
        kappa = 0.1

        # Data structure to hold the regrets obtained from each type of ACQ function in each run of the BO
        total_ucb_regret = []
        total_ei_regret = []
        total_pi_regret = []
        total_rs_regret = []

        # Added to generate the results for comparing regrets for spatially varying length scales.
        ei_ard_regret = []
        ei_var_l_regret = []
        ei_fixed_l_regret = []
        ei_multi_l_regret = []



        kernel_iter_types = ['var_l', 'm_ker', 'ard', 'fix_l']
        # kernel_iter_types = ['ard']
        # kernel_iter_types = ['m_ker']

        print("\n\nParameters\nTrue Func", true_func, "\tDim:", number_of_dimensions, "\tobs_samp:", number_of_observed_samples,
              "\titer:",
              number_of_iterations, "\tkernel_types:", kernel_iter_types,"\tlenscale_fun_types:",len_scale_func_type,"\tseed:", random_seed,
              "\nRuns:", number_of_runs, "\tAcq_re:", number_of_restarts_acq, "\tlikeli_re:", number_of_restarts_likelihood,
              "\tBounds:", bounds, "\tl_params:", len_scale_params, "\nl_param_bds:", len_scale_param_bounds,"\tard_len_scale_bds:",
              len_scale_bounds,
              "\nxmax:", Xmax, "\txmin:", Xmin, "\tymax:", ymax, "\tymin:", ymin, "\teps1:", epsilon1, "\teps2:", epsilon2,
              "\tfixed_l:", char_length_scale, "\tSig_var:", signal_variance, "\tSig_var_bds:", signal_variance_bounds, "\nTest_points:",
              number_of_test_datapoints,"\tlen_weights: ",len_weights,"\tmulti_len_scales", multi_len_scales,"\n\n")


        print("\n###################################################################\n")
        timenow = datetime.datetime.now()
        print("Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for i in range(number_of_runs):

            # Create Gaussian Process  object to carry on the prediction and model fitting tasks
            gaussianObject = GaussianProcess(kernel_type, params_estimation, len_scale_estimation, char_length_scale,
                                             len_scale_bounds, signal_variance, signal_variance_bounds,
                                             number_of_test_datapoints, noise,
                                             random_seed, linspacexmin, linspacexmax, linspaceymin, linspaceymax,
                                             bounds,
                                             number_of_dimensions, number_of_observed_samples, kernel_char,
                                             len_scale_params, len_scale_param_bounds, len_scale_func_type, number_of_restarts_likelihood,
                                             Xmin, Xmax, ymin, ymax, len_weight_bounds, len_weights, weights_estimation, multi_len_scales)

            # Creating an ACQ object to hold the parameters of the ACQ functions
            acq_func_obj = AcquisitionFunction(None, number_of_restarts_acq, kappa, epsilon1,
                                               epsilon2)

            func_helper_obj = FunctionHelper(true_func)

            # Create a Bayesian Optimization object to hold the parameters of the current Bayesian Optimization settings
            bay_opt_obj = BayesianOptimization('mybayesianobject', gaussianObject, acq_func_obj, func_helper_obj,
                                               number_of_iterations)

            # Generate the random sample points and its true function values, that acts as a prior to our GP

            # Data structure to hold the data
            random_points = []
            X = []

            # Generate specified (number of observed samples) random numbers for each dimension
            for dim in np.arange(number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                               number_of_observed_samples).reshape(1,
                                                                                                   number_of_observed_samples)
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

            # Obtain the function values observed at given randomly generated samples
            y = bay_opt_obj.func_helper_obj.get_true_func_value(X)
            true_max = func_helper_obj.get_true_max()

            # ##Normalizing code
            X_norm = np.divide((X-Xmin), (Xmax-Xmin))
            y_norm = (y-ymin)/(ymax-ymin)
            true_max_norm =  (true_max -ymin)/(ymax - ymin)
            bay_opt_obj.y_true_max = true_max_norm
            init_regret = []
            for index_norm in np.arange(len(y_norm)):
                init_regret.append(true_max_norm - max(y_norm[:index_norm + 1]))
            # init_regret = []
            # for index_norm in np.arange(len(y)):
            #     init_regret.append(true_max - max(y[:index_norm + 1]))

            print ("Regret for initial points: ",init_regret)
            # Fit the gaussian model for the generated sample data
            # gaussianObject.gaussian_fit(X, y)

            # Running the optimization for different acquisition values
            # Make sure to reset the Gaussian Process to the initial setting before running the optimization for
            # different acquisition functions, so that comparison can be made
            # Acquisi tion function types : 'ei', 'ucb', 'pi', 'rs'

            # #Algorithm running for PI acquisition function
            if ('pi' in acq_fun_list):
                acq_type = 'pi'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)
                # Resetting the GP model because of the above stated reason
                gaussianObject.char_length_scale = char_length_scale
                gaussianObject.signal_variance = signal_variance
                gaussianObject.gaussian_fit(X, y)

                # Store the regret obtained in each run so that mean and variance can be calculated to plot the simple regret
                pi_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                pi_regret_eachrun = np.append(init_regret, pi_regret_eachrun)
                total_pi_regret.append(pi_regret_eachrun)

            if ('rs' in acq_fun_list):
                # Algorithm running for random search
                acq_type = 'rs'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)
                gaussianObject.char_length_scale = char_length_scale
                gaussianObject.signal_variance = signal_variance
                gaussianObject.gaussian_fit(X, y)
                rs_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                rs_regret_eachrun = np.append(init_regret, rs_regret_eachrun)
                total_rs_regret.append(rs_regret_eachrun)

            if ('ei' in acq_fun_list):
                # Algorithm running for EI acquisition function
                acq_type = 'ei'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)

                for iter_type in kernel_iter_types:
                    gaussianObject.char_length_scale = char_length_scale
                    gaussianObject.signal_variance = signal_variance
                    gaussianObject.len_weights = len_weights
                    gaussianObject.len_scale_params = len_scale_params
                    gaussianObject.kernel_char = iter_type
                    gaussianObject.gaussian_fit(X_norm, y_norm)
                    # gaussianObject.gaussian_fit(X, y)
                    # total_ei_regret.append(ei_regret_eachrun)         # Commented as it is replaced by other code

                    if(iter_type == 'ard'):
                        print("Testing with ARD Kernel in Run: ", (i+1))
                        gaussianObject.len_scale_estimation = True
                        ei_regret_eachrun =  bay_opt_obj.run_bayes_opt(i + 1)
                        ei_regret_eachrun = np.append(init_regret, ei_regret_eachrun)
                        gaussianObject.len_scale_estimation = False
                        ei_ard_regret.append(ei_regret_eachrun)

                    elif(iter_type == 'var_l'):
                        print("Testing with Varying Kernel in Run: ", (i + 1))
                        gaussianObject.params_estimation = True
                        ei_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                        ei_regret_eachrun = np.append(init_regret, ei_regret_eachrun)
                        gaussianObject.params_estimation = False
                        ei_var_l_regret.append(ei_regret_eachrun)

                    elif (iter_type == 'fix_l'):
                        print("Testing with fixed Kernel in Run: ", (i + 1))
                        gaussianObject.params_estimation = False
                        gaussianObject.len_scale_estimation = False
                        gaussianObject.weights_estimation = False
                        ei_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                        ei_regret_eachrun = np.append(init_regret, ei_regret_eachrun)
                        ei_fixed_l_regret.append(ei_regret_eachrun)

                    elif (iter_type == 'm_ker'):
                        print("Testing with multi Kernel in Run: ", (i + 1))
                        gaussianObject.weights_estimation =True
                        ei_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                        ei_regret_eachrun = np.append(init_regret, ei_regret_eachrun)
                        gaussianObject.weights_estimation = False
                        ei_multi_l_regret.append(ei_regret_eachrun)


            if ('ucb' in acq_fun_list):
                # Algorithm running for UCB acquisition function
                acq_type = 'ucb'
                bay_opt_obj.acq_func_obj.set_acq_func_type(acq_type)
                gaussianObject.char_length_scale = char_length_scale
                gaussianObject.signal_variance = signal_variance
                gaussianObject.gaussian_fit(X, y)
                ucb_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                ucb_regret_eachrun = np.append(init_regret, ucb_regret_eachrun)
                total_ucb_regret.append(ucb_regret_eachrun)

            # plot regret to verify the posterior after each run
            # iterations_axes = np.arange(start = 1, stop = number_of_iterations+1, step = 1)
            # plt.figure("Regret"+str(i+1))
            # plt.clf()
            # plt.plot(iterations_axes, pi_regret_eachrun, 'b')
            # plt.axis([1, number_of_iterations, 0 , 1])
            # plt.title('Regret for run: '+ str(i))
            # plt.savefig('regret_ucb'+str(i)+'.png')

            # $$$$$$$$$$$$$$$$$$$$$$$$        End of each run       $$$$$$$$$$$$$$$$$$$$$$$$$

        # Generate the values for the Xaxis in the simple regret graph to display mean and variance at each evaluation
        # iterations_axes_values = np.arange(start = 1, stop = number_of_iterations+1, step = 1)
        # Changed from i+1 to i+3 as now the regret includes regret of inital points as well
        iterations_axes_values = [i + 1 for i in np.arange(number_of_iterations+(number_of_dimensions+1))]

        fig_name = 'Regret_'+str(true_func)+'_iter'+str(number_of_iterations)+'_runs'+str(number_of_runs)+'_'
        plt.figure(str(fig_name))
        plt.clf()

        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if ('pi' in acq_fun_list):
            # Accumulated values for the regret is used to calculate the mean and deviations at each evaluations/iterations
            # Vertically stack the arrays obtained in each run as a matrix rows and columns to calculate the mean and deviations
            total_pi_regret = np.vstack(total_pi_regret)

            # Calculate the mean and deviations of regrets for each evaluations in each run
            pi_regret_mean = np.mean(total_pi_regret, axis=0)
            pi_regret_std_dev = np.std(total_pi_regret, axis=0)

            # Display the values of the regret to help debugging if required
            print("\n\nTotal PI Regret\n", total_pi_regret, "\n\nPI Regret Mean",
                  pi_regret_mean, "\n\nPI Regret Deviation\n", pi_regret_std_dev)

            # Plot the deviations observed for each of the ACQ type and errorbar if required
            ax.plot(iterations_axes_values, pi_regret_mean, 'g')
            # plt.errorbar(iterations_axes_values, pi_regret_mean, yerr=pi_regret_std_dev)
            plt.gca().fill_between(iterations_axes_values, pi_regret_mean + pi_regret_std_dev,
                                   pi_regret_mean - pi_regret_std_dev, color="green", alpha=0.25, label='PI')


        if ('ei' in acq_fun_list):

            #Commented to plot graphs for specific type of kernel using the same EI acq.
            # total_ei_regret = np.vstack(total_ei_regret)
            # ei_regret_mean = np.mean(total_ei_regret, axis=0)
            # ei_regret_std_dev = np.std(total_ei_regret, axis=0)
            # print("\n\nTotal EI Regret\n", total_ei_regret, "\n\nEI Regret Mean", ei_regret_mean,
            #       "\n\nEI Regret Deviation\n",
            #       ei_regret_std_dev)
            #
            # ax.plot(iterations_axes_values, ei_regret_mean, 'b')
            # # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            # plt.gca().fill_between(iterations_axes_values, ei_regret_mean + ei_regret_std_dev,
            #                        ei_regret_mean - ei_regret_std_dev, color="blue", alpha=0.25, label="EI")


            # Modifications here to accomodate new variables to plot
            for iter_type in kernel_iter_types:
                if (iter_type == 'ard'):
                    ei_ard_regret = np.vstack(ei_ard_regret)
                    ei_ard_regret_mean = np.mean(ei_ard_regret, axis=0)
                    ei_ard_regret_std_dev = np.std(ei_ard_regret, axis=0)
                    ei_ard_regret_std_dev= ei_ard_regret_std_dev/ np.sqrt(number_of_iterations + (number_of_dimensions + 1))
                    print("\n\nTotal EI ARD Regret\n", ei_ard_regret, "\n\nARD Regret Mean", ei_ard_regret_mean,
                          "\n\nARD Regret Deviation_Std_error\n", ei_ard_regret_std_dev)

                    ax.plot(iterations_axes_values, ei_ard_regret_mean, 'b', linestyle='dashed')
                    # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
                    plt.gca().fill_between(iterations_axes_values, ei_ard_regret_mean + ei_ard_regret_std_dev,
                                ei_ard_regret_mean - ei_ard_regret_std_dev, color="blue", alpha=0.25, label="ARD")


                elif (iter_type == 'var_l'):
                    ei_var_l_regret = np.vstack(ei_var_l_regret)
                    ei_var_regret_mean = np.mean(ei_var_l_regret, axis=0)
                    ei_var_regret_std_dev = np.std(ei_var_l_regret, axis=0)
                    ei_var_regret_std_dev=ei_var_regret_std_dev/ np.sqrt(number_of_iterations + (number_of_dimensions + 1))
                    print("\n\nTotal EI VAR Regret\n", ei_var_l_regret, "\n\nVAR Regret Mean", ei_var_regret_mean,
                          "\n\nVAR Regret Deviation_std_error\n", ei_var_regret_std_dev)

                    ax.plot(iterations_axes_values, ei_var_regret_mean, 'g', linestyle='solid')
                    # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
                    plt.gca().fill_between(iterations_axes_values, ei_var_regret_mean + ei_var_regret_std_dev,
                                           ei_var_regret_mean - ei_var_regret_std_dev, color="green", alpha=0.25,
                                           label="Spatial")


                elif (iter_type == 'fix_l'):
                    ei_fixed_l_regret = np.vstack(ei_fixed_l_regret)
                    ei_fix_regret_mean = np.mean(ei_fixed_l_regret, axis=0)
                    ei_fix_regret_std_dev = np.std(ei_fixed_l_regret, axis=0)
                    ei_fix_regret_std_dev= ei_fix_regret_std_dev/ np.sqrt(number_of_iterations + (number_of_dimensions + 1))
                    print("\n\nTotal EI FIX Regret\n", ei_fixed_l_regret, "\n\nFIX Regret Mean", ei_fix_regret_mean,
                          "\n\nFIX Regret Deviation_std_error\n", ei_fix_regret_std_dev)

                    ax.plot(iterations_axes_values, ei_fix_regret_mean, 'r', linestyle='dashdot')
                    # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
                    plt.gca().fill_between(iterations_axes_values, ei_fix_regret_mean + ei_fix_regret_std_dev,
                                           ei_fix_regret_mean - ei_fix_regret_std_dev, color="red", alpha=0.25,
                                           label="Fixed ")

                elif (iter_type == 'm_ker'):
                    ei_multi_l_regret = np.vstack(ei_multi_l_regret)
                    ei_multi_regret_mean = np.mean(ei_multi_l_regret, axis=0)
                    ei_multi_regret_std_dev = np.std(ei_multi_l_regret, axis=0)
                    ei_multi_regret_std_dev= ei_multi_regret_std_dev/ np.sqrt(number_of_iterations + (number_of_dimensions + 1))
                    print("\n\nTotal EI FIX Regret\n", ei_multi_l_regret, "\n\nFIX Regret Mean", ei_multi_regret_mean,
                          "\n\nFIX Regret Deviation_std_error\n", ei_multi_regret_std_dev)

                    ax.plot(iterations_axes_values, ei_multi_regret_mean, 'k', linestyle='dotted')
                    # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
                    plt.gca().fill_between(iterations_axes_values, ei_multi_regret_mean + ei_multi_regret_std_dev,
                                           ei_multi_regret_mean - ei_multi_regret_std_dev, color="grey", alpha=0.25,
                                           label="Multi")

        if ('ucb' in acq_fun_list):
            total_ucb_regret = np.vstack(total_ucb_regret)
            ucb_regret_mean = np.mean(total_ucb_regret, axis=0)
            ucb_regret_std_dev = np.std(total_ucb_regret, axis=0)
            print("\n\nTotal UCB Regret\n", total_ucb_regret, "\n\nUCB Regret Mean", ucb_regret_mean,
                  "\n\nUCB Regret Deviation\n", ucb_regret_std_dev)

            ax.plot(iterations_axes_values, ucb_regret_mean, 'r')
            # plt.errorbar(iterations_axes_values, ucb_regret_mean, yerr=ucb_regret_std_dev)
            plt.gca().fill_between(iterations_axes_values, ucb_regret_mean + ucb_regret_std_dev,
                                   ucb_regret_mean - ucb_regret_std_dev, color="red", alpha=0.25, label='UCB')


        if ('rs' in acq_fun_list):
            total_rs_regret = np.vstack(total_rs_regret)
            rs_regret_mean = np.mean(total_rs_regret, axis=0)
            rs_regret_std_dev = np.std(total_rs_regret, axis=0)
            print("\n\nTotal RS Regret\n", total_rs_regret, "\n\nRS Regret Mean", rs_regret_mean,
                  "\n\nRS Regret Deviation\n", rs_regret_std_dev)

            ax.plot(iterations_axes_values, rs_regret_mean, 'y')
            # plt.errorbar(iterations_axes_values, ucb_regret_mean, yerr=ucb_regret_std_dev)
            plt.gca().fill_between(iterations_axes_values, rs_regret_mean + rs_regret_std_dev,
                                   rs_regret_mean - rs_regret_std_dev, color="yellow", alpha=0.25, label='RS')

        timenow = datetime.datetime.now()
        print("\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Set the parameters of the simple regret graph
        plt.title('Regret')
        plt.xlabel('#Evaluations')
        plt.ylabel('Simple Regret')
        plt.savefig(fig_name+str(start_time)+'.png')
        # plt.axis([1, len(iterations_axes_values), 0, 0.5])
        # legend = ax.legend(loc=1, fontsize='x-small')
        label = ["FIX", "ARD", "VAR", "MULTI"]
        plt.xlim(1, len(iterations_axes_values))
        custom_lines = [Line2D([0], [0], lw=1, ls="dashdot", color='red'),
                        Line2D([0], [0], lw=1, ls="dashed", color='blue'),
                        Line2D([0], [0], lw=1, ls="solid", color='green'),
                        Line2D([0], [0], lw=1, ls="dotted", color='black')
                        ]
        ax.legend(custom_lines, label, loc=1, fontsize='x-small')

        plt.show()



    if __name__ == "__main__":
        timenow = datetime.datetime.now()
        stamp =  timenow.strftime("%H%M%S_%d%m%Y")
        f = open('console_output_'+str(stamp)+'.txt', 'w')
        original = sys.stdout
        sys.stdout = Custom_Print(sys.stdout, f)
        opt_wrapper(stamp)




'''
#################### References ###########################

#different ways of generating data 


# Fixed samples
# X = np.array([[1], [3], [4], [8]]).reshape(4, 1)

# Randomly select samples between the upper bound and the lower bound
# X = np.array(np.random.uniform(linspacexmin, linspacexmax,
#                                number_of_observed_samples)).reshape(number_of_observed_samples, 1)


random_points = []
X = []

for dim in np.arange(number_of_dimensions):
    random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                   number_of_observed_samples).reshape(1, number_of_observed_samples)
    print(random_data_point_each_dim)
    random_points.append(random_data_point_each_dim)

random_points = np.vstack(random_points)

for sample_num in np.arange(number_of_observed_samples):
    array = []
    for dim_count in np.arange(number_of_dimensions):
        array.append(random_points[dim_count, sample_num])
    X.append(array)
X = np.vstack(X)


        # Doesnt work for multi dimensions
        # X =[]
        # for sample_num in np.arange(number_of_observed_samples):
        #     array =[]
        #     for dim_count in np.arange(number_of_dimensions):
        #         value =  np.random.uniform(bounds[dim_count][0], bounds[dim_count][1],1)
        #         array.append(value)
        #     X.append(array)
        #
        # X = np.vstack(X)

### Tried different methods to append the matrix

            # ynew = self.func_helper_obj.get_true_func_value(np.matrix(xnew))
            # y = np.append(self.gp_obj.y, ynew.reshape(-1,self.gp_obj.number_of_dimensions), axis=0)
            # self.gp_obj.gaussian_fit(X, y)

            # X = self.gp_obj.X
            # if( np.array_equal(xnew,np.zeros(self.gp_obj.number_of_dimensions))):
            #     zero_value_bool = True
            #     print('\nzeroes encountered in run: ', run_count, " iteration: ", i + 1)
            #
            # xnew = np.matrix(xnew)
            # ynew = self.func_helper_obj.get_true_func_value(xnew)
            # X = np.append(X, xnew, axis=0)
            #
            # # ynew = np.array(ynew).ravel()
            # y = np.append(self.gp_obj.y, ynew.reshape(-1,self.gp_obj.number_of_dimensions), axis=0)
            # self.gp_obj.gaussian_fit(X, y)
            #
            # print("added value ", X.shape, y.shape)


            # plot_axes = [linspacexmin, linspacexmax, 0, 20]
            # self.acq_func_obj.plot_acquisition_function(i+1, Xs, acq_func_values, plot_axes)



'''