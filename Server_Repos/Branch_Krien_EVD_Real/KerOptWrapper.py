from HyperGaussianProcess import HyperGaussianProcess
from AcquisitionUtility import AcquisitionUtility
from KernelOptimiser import KernelOptimiser
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys
sys.path.insert(0, 'C:\Arun_Stuff\GitHubCodes\KForm\HyperKernelBayesianOptimisation\HyperKernelFunctionalOptimisation\Branch_Krien_EVD_Real'
                   '\GP_Regressor')
from GaussianProcessWrapper import GaussianProcessWrapper
from Functions import FunctionHelper

import os
sys.path.append("..")
from HelperUtility.PrintHelper import PrintHelper as PH


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
class KernelOptimizationWrapper:

    def kernel_wrapper(self, start_time, input):

        # Number of data points in X  for the grid
        number_of_samples_in_X_for_grid = 4
        number_of_samples_in_Xs_for_grid = 5

        # Lower bound considered, required for sample generation
        min_X = 0
        # Upper bound considered, required for sample generation
        max_X = 1
        # Minimum value for the objective function
        min_Y = 1
        # Maximum value for the objective function
        max_Y = 2

        number_of_dimensions = 1
        bounds = [[min_X, max_X] for d in range(number_of_dimensions)]

        # Number of points required to be observed to evaluate the unknown function ((10:20)*no_dimensions )
        # number_of_iterations = number_of_dimensions * 10
        number_of_subspace_selection_iterations = 5
        number_of_iterations_best_solution = 5

        # LINEBO
        # number_of_basis_vectors_chosen = 2
        number_of_basis_vectors_chosen = 1

        basis_weights_bounds = [0.1, 1]

        number_of_init_random_kernel_y_observations = 4

        # extrema_type = -1 if function maxima is considered
        # extrema_type = 1  if function minima is considered Ex: Regret is +infinity, as Regret reduces over iterations

        #Commenting to modify the code for calculating the minimum negative likelihood
        # extrema_type = -1
        extrema_type = 1

        # Number of samples for which we have the y values to be known (no_dimensions+1)
        # number_of_observed_samples = number_of_dimensions + 1

        # Number of restarts required during the calculation of the maxima during the maximization of acq functions
        number_of_restarts_acq = 100

        # Number of restarts required during the calculation of the maxima during the maximization of likelihood
        number_of_restarts_likelihood = 100

        # Number of runs the BO has to be run for calculating the regret and the corresponding means and the standard devs
        number_of_runs = 2

        # Type of kernel to be used in the optimization process
        # 0 - Squared Exponential; 1 - Rational Quadratic Function; 2 - Exponential; 3 - Periodic***(To be fixed)
        # hyperkernel_type = 'gaussian_harmonic_kernel'
        hyperkernel_type = 'matern_harmonic_kernel'
        # hyperkernel_type = 'polynomial_kernel'
        # hyperkernel_type = 'free_four_kernel'
        # hyperkernel_type = 'free_rbf_kernel'
        # hyperkernel_type = 'linear_kernel'

        #exp1 Linsin 27
        # hyper_lambda = 0.13733357
        #original
        # hyper_lambda = 0.01
        #exp linsin 28
        hyper_lambda = 0.1

        # Matern Harmonic Kernel
        # hyper_char_len_scale = 0.43114668
        # hyper_char_len_scale = 0.105
        hyper_char_len_scale = 0.1

        if input is not None:
            hyper_lambda = input[0]
            hyper_char_len_scale = input[1]
            PH.printme(PH.p1, "Inputs Supplied: ", input)

        # Free RBF
        # hyper_char_len_scale = 1.2


        # Characteristics of the Kernel being used, currently implemented for the square exponential kernel
        # Required to set up the GP Object initially
        kernel_char = 'ard'

        # characteristic_length_scale = np.array([1 for nd in np.arange(number_of_dimensions)])
        # char_length_scale = [1 for nd in np.arange(number_of_dimensions)]
        char_length_scale = 0.3
        len_scale_bounds = [[0.1, 1]]

        # sigma_len = 0.0001
        sigma_len = 0.0001
        sigma_len_bounds = [0.1, 1]

        # Signal variance bounds
        signal_variance_bounds = [0.1, 1]

        # Initial Signal Variance
        signal_variance = 1

        # Number of unseen data points in order to evaluate the function
        number_of_test_datapoints = 100

        # Noise to be added in the system***
        noise = 0.0

        # Number of principal components
        no_principal_components = 10

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

        # kernel_iter_types = ['var_l', 'm_ker', 'ard', 'fix_l']
        kernel_iter_types = ['fix_l']
        # kernel_iter_types = ['m_ker']

        tot_min_loglik = np.array([])
        tot_best_sol = np.array([])

        kernel_type = "SE"

        # synthetic_data = True
        # dataset = None
        synthetic_data = False
        # dataset = 'challenger'
        # dataset = 'concrete'
        dataset = 'fertility'
        # dataset = 'yacht'
        # dataset = 'boston'
        # dataset = 'airfoil'

        # Initial value to denote the type of ACQ function to be used, but ignored as all ACQs are run in the sequence
        # acq_fun_list = ['ei', 'pi', 'rs', 'ucb']
        acq_fun_list = ['UCB']

        PH.printme(PH.p1, "Configuration Settings:\n\n", "hyperkernel_type:", hyperkernel_type, "\tkernel_type:", kernel_type,
                   "\tchar_length_scale:", char_length_scale, "\tsigma_len:", sigma_len, "\tsigma_len_bounds: ", sigma_len_bounds,
                   "\tsignal_variance_bounds:", signal_variance_bounds, "\nnumber_of_samples_in_X_for_grid:",
                   number_of_samples_in_X_for_grid, "\tnumber_of_samples_in_Xs_for_grid:", number_of_samples_in_Xs_for_grid,
                   "\tnumber_of_test_datapoints:", number_of_test_datapoints, "\tnoise:", noise, "\nhyper_lambda", hyper_lambda,
                   "\trandom_seed:", random_seed, "\tmax_X:", max_X, "\tmin_X:", min_X, "\tmax_Y:", max_Y, "\tmin_Y:", min_Y,
                   "\nbounds:", bounds, "\tnumber_of_dimensions:", number_of_dimensions, "\nsignal_variance:", signal_variance,
                   "\tnumber_of_basis_vectors_chosen: ", number_of_basis_vectors_chosen, "\tbasis_weights_bounds:", basis_weights_bounds,
                   "\nnumber_of_subspace_selection_iterations:", number_of_subspace_selection_iterations,
                   "\tnumber_of_iterations_best_solution:", number_of_iterations_best_solution,
                   "\nnumber_of_init_random_kernel_y_observations:", number_of_init_random_kernel_y_observations, "\tacq_fun_list:",
                   acq_fun_list, "\nLength scale bounds: ", len_scale_bounds, "\tnumber of restart likelihoods: ",
                   number_of_restarts_likelihood, "\thyper_char_len_scale: ", hyper_char_len_scale)

        if not synthetic_data:
            PH.printme(PH.p1, "Dataset: ", dataset)


        PH.printme(PH.p1, "\n###################################################################\n")
        timenow = datetime.datetime.now()
        PH.printme(PH.p1, "Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for i in range(number_of_runs):

            X = []
            # X = np.linspace(X_space_min, X_space_max, number_of_samples_in_X).reshape(number_of_samples_in_X, number_of_dimensions)

            # updated implementation
            random_points = []
            # Generate specified (number of observed samples) random numbers for each dimension
            for dim in np.arange(number_of_dimensions):
                # random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1], number_of_observed_samples).reshape(1,
                #                                                                                     number_of_observed_samples)

                random_data_point_each_dim = np.linspace(min_X, max_X, number_of_samples_in_X_for_grid).reshape(1,
                                                                                                        number_of_samples_in_X_for_grid)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
            random_points = np.vstack(random_points)

            # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
            for sample_num in np.arange(number_of_samples_in_X_for_grid):
                array = []
                for dim_count in np.arange(number_of_dimensions):
                    array.append(random_points[dim_count, sample_num])
                X.append(array)
            X = np.vstack(X)

            # Normalisation code
            # X = np.divide((X - min_X), (max_X - min_X))

            # Create Gaussian Process  object to carry on the prediction and model fitting tasks
            hyper_gaussian_object = HyperGaussianProcess(X, hyperkernel_type, kernel_type, char_length_scale, sigma_len, sigma_len_bounds,
                                                         signal_variance_bounds, number_of_samples_in_X_for_grid,
                                                         number_of_samples_in_Xs_for_grid,
                                                         number_of_test_datapoints, noise, hyper_lambda,
                                                         random_seed, max_X, min_X, max_Y, min_Y, bounds,
                                                         number_of_dimensions, signal_variance, number_of_basis_vectors_chosen,
                                                         basis_weights_bounds, len_scale_bounds, number_of_restarts_likelihood,
                                                         no_principal_components, hyper_char_len_scale)

            acquisition_utility_object = AcquisitionUtility(None, number_of_restarts_acq, extrema_type)

            gp_wrapper_obj = GaussianProcessWrapper(synthetic_data)

            if synthetic_data:
                posterior_plot = True
                gp_wrapper_obj.construct_gp_regressor_syn(start_time, posterior_plot)

            else:
                posterior_plot = False
                gp_wrapper_obj.construct_gp_regressor_real(start_time, dataset, posterior_plot)

            kernel_optimiser_obj = KernelOptimiser(hyper_gaussian_object, acquisition_utility_object,
                                                   number_of_subspace_selection_iterations, number_of_iterations_best_solution,
                                                   number_of_init_random_kernel_y_observations, gp_wrapper_obj)

            # #Algorithm running for PI acquisition function
            if 'UCB' in acq_fun_list:
                acq_type = 'UCB'
                kernel_optimiser_obj.acquisition_utility_object.set_acq_func_type(acq_type)
                best_solution_found = kernel_optimiser_obj.optimise_kernel(i + 1)
                PH.printme(PH.p1, "Best solution found for run: ", i+1, kernel_optimiser_obj.best_solution)
                minimum_likelihood = gp_wrapper_obj.compute_likelihood_for_kernel('HYPER', best_solution_found['best_kernel'],
                                                                                  hyper_gaussian_object)
                PH.printme(PH.p1, "Maximum log marginal likelihood obtained: ", minimum_likelihood)
                tot_min_loglik = np.append(tot_min_loglik, minimum_likelihood)
                tot_best_sol = np.append(tot_best_sol, best_solution_found)
                PH.printme(PH.p1, "\n\n***************************Run ", i+1, " completed**********************\n\n\n\n")


        # # #Commenting the following code for integration with Outer Bayesian Optimisation
        # # # PH.printme(PH.p1, "Initiating kernel plotting...")
        # # # gp_wrapper_obj.plot_GP_Reg__kernel("Kernel Learnt", best_solution_found['best_kernel'])
        # PH.printme(PH.p1, "Plotting posterior distribution")
        # gp_wrapper_obj.compute_posterior_distribution('HYPER', best_solution_found['best_kernel'], hyper_gaussian_object, "final posterior")

        mean_min_likelihood = np.mean(tot_min_loglik)

        PH.printme(PH.p1, "\n\n@@@@@@@Mean likelihood for Bayesian Optimisation : ", mean_min_likelihood)
        # # Uncomment to see the graph if not integrated with BO
        # plt.show()
        return mean_min_likelihood

                # Store the regret obtained in each run so that mean and variance can be calculated to plot the simple regret
                # pi_regret_eachrun = bay_opt_obj.run_bayes_opt(i + 1)
                # pi_regret_eachrun = np.append(init_regret, pi_regret_eachrun)
                # total_pi_regret.append(pi_regret_eachrun)

            # for kernel in kernels_list:
            #     lik = gp_wrapper_obj.compute_likelihood_for_kernel(kernel)


        # iterations_axes_values = [i + 1 for i in np.arange(number_of_iterations+(number_of_dimensions+1))]
        # fig_name = 'Regret_'+'_iter'+str(number_of_iterations)+'_runs'+str(number_of_runs)+'_'
        # plt.figure(str(fig_name))
        # plt.clf()
        #
        # ax = plt.subplot(111)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #
        # if ('pi' in acq_fun_list):
        #     # Accumulated values for the regret is used to calculate the mean and deviations at each evaluations/iterations
        #     # Vertically stack the arrays obtained in each run as a matrix rows and columns to calculate the mean and deviations
        #     total_pi_regret = np.vstack(total_pi_regret)
        #
        #     # Calculate the mean and deviations of regrets for each evaluations in each run
        #     pi_regret_mean = np.mean(total_pi_regret, axis=0)
        #     pi_regret_std_dev = np.std(total_pi_regret, axis=0)
        #
        #     # Display the values of the regret to help debugging if required
        #     PH.printme(PH.p1, "\n\nTotal PI Regret\n", total_pi_regret, "\n\nPI Regret Mean",
        #           pi_regret_mean, "\n\nPI Regret Deviation\n", pi_regret_std_dev)
        #
        #     # Plot the deviations observed for each of the ACQ type and errorbar if required
        #     ax.plot(iterations_axes_values, pi_regret_mean, 'g')
        #     # plt.errorbar(iterations_axes_values, pi_regret_mean, yerr=pi_regret_std_dev)
        #     plt.gca().fill_between(iterations_axes_values, pi_regret_mean + pi_regret_std_dev,
        #                            pi_regret_mean - pi_regret_std_dev, color="green", alpha=0.25, label='PI')
        #
        #
        #  if ('ucb' in acq_fun_list):
        #     total_ucb_regret = np.vstack(total_ucb_regret)
        #     ucb_regret_mean = np.mean(total_ucb_regret, axis=0)
        #     ucb_regret_std_dev = np.std(total_ucb_regret, axis=0)
        #     PH.printme(PH.p1, "\n\nTotal UCB Regret\n", total_ucb_regret, "\n\nUCB Regret Mean", ucb_regret_mean,
        #           "\n\nUCB Regret Deviation\n", ucb_regret_std_dev)
        #
        #     ax.plot(iterations_axes_values, ucb_regret_mean, 'r')
        #     # plt.errorbar(iterations_axes_values, ucb_regret_mean, yerr=ucb_regret_std_dev)
        #     plt.gca().fill_between(iterations_axes_values, ucb_regret_mean + ucb_regret_std_dev,
        #                            ucb_regret_mean - ucb_regret_std_dev, color="red", alpha=0.25, label='UCB')
        #
        # timenow = datetime.datetime.now()
        # PH.printme(PH.p1, "\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))
        #
        # # Set the parameters of the simple regret graph
        # plt.axis([1, len(iterations_axes_values), 0, 0.5])
        # plt.title('Regret')
        # plt.xlabel('Evaluations')
        # plt.ylabel('Simple Regret')
        # plt.savefig(fig_name+str(start_time)+'.png')
        # legend = ax.legend(loc=1, fontsize='x-small')
        # plt.show()


if __name__ == "__main__":
    timenow = datetime.datetime.now()
    stamp = timenow.strftime("%H%M%S_%d%m%Y")
    PH(os.getcwd())
    input = None
    ker_opt_wrapper_obj = KernelOptimizationWrapper()
    ker_opt_wrapper_obj.kernel_wrapper(stamp, input)
