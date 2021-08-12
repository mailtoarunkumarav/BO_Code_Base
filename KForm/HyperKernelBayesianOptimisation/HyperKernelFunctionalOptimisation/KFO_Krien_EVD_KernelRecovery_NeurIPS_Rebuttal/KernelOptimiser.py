import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from HelperUtility.PrintHelper import PrintHelper as PH



class KernelOptimiser:

    def __init__(self, hyper_gaussian_object, acquisition_utility_object, number_of_subspace_selection_iterations,
                 number_of_iterations_best_solution, number_of_init_random_kernel_y_observations, gp_wrapper_obj):
        self.hyper_gaussian_object = hyper_gaussian_object
        self.acquisition_utility_object = acquisition_utility_object
        self.number_of_subspace_selection_iterations = number_of_subspace_selection_iterations
        self.number_of_iterations_best_solution = number_of_iterations_best_solution
        self.number_of_init_random_kernel_y_observations = number_of_init_random_kernel_y_observations
        self.best_solution = {}
        self.gp_wrapper_obj = gp_wrapper_obj
        self.complete_dataset_kernel_observations = np.array([]).reshape(-1, self.hyper_gaussian_object.no_principal_components)
        self.complete_dataset_y = np.array([]).reshape(-1, 1)

    def generate_observations(self, kernel_bias, basis_weights, kernel_samples):

        # Generate initial observations for the kernel optimisation with the selected basis
        #LINEBO
        # observations_kernel = basis_weights[0] * kernel_bias + basis_weights[1] * kernel_samples[0] + basis_weights[2] * kernel_samples[1]
        observations_kernel = kernel_bias + basis_weights[0] * kernel_samples[0]

        # Process for running observations
        # observations_y = run_BO_for_kernel(observations_kernel)
        observations_y = self.gp_wrapper_obj.compute_likelihood_for_kernel('HYPER', observations_kernel, self.hyper_gaussian_object)
        # observations_y = np.random.uniform(0, 1, 1).reshape(1, -1)
        return observations_kernel, observations_y

    def sample_basis_weights(self):
        # Generate weights to the selected basis vectors
        weights = np.random.uniform(self.hyper_gaussian_object.basis_weights_bounds[0], self.hyper_gaussian_object.basis_weights_bounds[
            1], self.hyper_gaussian_object.number_of_basis_vectors_chosen
        ).reshape(-1, 1)
        return weights

    def optimise_kernel(self, run_count):

        # print the parameters for the current run
        PH.printme(PH.p1, "************  Run: " + str(run_count) + "    ***************\n\n")
        PH.printme(PH.p1, 'Initial Grid Values selected for this run\n--- X in Grid ---\n', self.hyper_gaussian_object.X)

        # Without kappa EVD
        # self.best_solution['best_kernel'] = np.zeros(shape=(np.square(self.hyper_gaussian_object.number_of_samples_in_X_for_grid),
        #                                                     1)).reshape(1, -1)

        # # Eigen based best solution
        self.best_solution['best_kernel'] = np.zeros(shape=(1, self.hyper_gaussian_object.no_principal_components))
        PH.printme(PH.p1, "Starting Optimization")

        bool_comp = False
        bool_kernel_plot = True
        # Run the optimization for the number of iterations and hence finding the best points to observe the function

        self.hyper_gaussian_object.compute_kappa_utils()

        for subspace_selection_count in range(self.number_of_subspace_selection_iterations):

            PH.printme(PH.p1, "######  run: " + str(run_count) + "  subspace ", subspace_selection_count + 1, "  ######")
            self.hyper_gaussian_object.current_kernel_samples = self.hyper_gaussian_object.generate_basis_as_kernel_samples()
            self.hyper_gaussian_object.current_kernel_bias = self.best_solution['best_kernel']

            for number_init_random_kernel_y_obs in range(self.number_of_init_random_kernel_y_observations):
                basis_weights = self.sample_basis_weights()
                observations_kernel, observations_y = self.generate_observations(self.hyper_gaussian_object.current_kernel_bias,
                                                                                 basis_weights,
                                                                                 self.hyper_gaussian_object.current_kernel_samples)
                if bool_comp:
                    PH.printme(PH.p1, "Plotting posterior before optimising kernel......")
                    self.gp_wrapper_obj.compute_posterior_distribution('HYPER', observations_kernel, self.hyper_gaussian_object,
                                                                       "initial posterior")
                    bool_comp = False

                self.hyper_gaussian_object.observations_kernel.append(observations_kernel)
                self.hyper_gaussian_object.observations_y = np.append(self.hyper_gaussian_object.observations_y, observations_y)
                PH.printme(PH.p1, "Generated new observation with observation_y: ", observations_y)

            # Vertical stacking for making it suitable for computation
            self.hyper_gaussian_object.observations_kernel = np.vstack(self.hyper_gaussian_object.observations_kernel)
            self.hyper_gaussian_object.observations_y = np.vstack(self.hyper_gaussian_object.observations_y)

            # Search for the best solution in the subspace selected
            for best_solution_count in range(self.number_of_iterations_best_solution):
                PH.printme(PH.p1, "\nTrying for best solution at iteration Number:", best_solution_count+1, "   in subspace: ",
                           (subspace_selection_count+1))
                PH.printme(PH.p1, "Using log marginal likelihood to compute the optimised length scale for the kernels")
                self.hyper_gaussian_object.compute_hyperparams_kernel_observations()
                PH.printme(PH.p1, "optimised values: l = ", self.hyper_gaussian_object.char_length_scale, "  sig = ",
                      self.hyper_gaussian_object.signal_variance)

                PH.printme(PH.p1, "pre-calculating COV matrix for kernel observations")
                cov_K_K_hypergp = self.hyper_gaussian_object.compute_covariance_matrix_for_kernels(
                    self.hyper_gaussian_object.observations_kernel, self.hyper_gaussian_object.observations_kernel)
                self.hyper_gaussian_object.L_K_K_hypergp = np.linalg.cholesky(cov_K_K_hypergp + 1e-6 * np.eye(len(
                    self.hyper_gaussian_object.observations_kernel)))

                best_basis_weights = self.acquisition_utility_object.maximise_acq_function(self.hyper_gaussian_object,
                                                                                           best_solution_count+1)
                PH.printme(PH.p1, "Best weights: ", best_basis_weights)
                observations_kernel_new, observations_y_new = self.generate_observations(self.hyper_gaussian_object.current_kernel_bias,
                                                                                         best_basis_weights,
                                                                                         self.hyper_gaussian_object.current_kernel_samples)
                PH.printme(PH.p1, "New observation found in the given subspace")
                PH.printme(PH.p1, "New kernel suggested: ", observations_kernel_new)
                PH.printme(PH.p1, "with observation value: ", observations_y_new)
                self.hyper_gaussian_object.observations_y = np.append(self.hyper_gaussian_object.observations_y, [observations_y_new[0]],
                                                                      axis=0)
                self.hyper_gaussian_object.observations_kernel = np.append(self.hyper_gaussian_object.observations_kernel,
                                                                           [observations_kernel_new[0]], axis=0)

                # Resetting the cached covariance matrix for kernel observations
                self.hyper_gaussian_object.L_K_K_hypergp = None

                # PH.printme(PH.p1, "after", self.hyper_gaussian_object.observations_y)

            PH.printme(PH.p1, "######    search in subspace", (subspace_selection_count + 1), " completed   ######")

            self.complete_dataset_y = np.append(self.complete_dataset_y, self.hyper_gaussian_object.observations_y, axis=0)
            self.complete_dataset_kernel_observations = np.append(self.complete_dataset_kernel_observations,
                                                                  self.hyper_gaussian_object.observations_kernel, axis=0)
            index_subspace_best_value = self.complete_dataset_y.argmax()
            self.best_solution['best_kernel'] = np.array(self.complete_dataset_kernel_observations[index_subspace_best_value]).reshape(1,
                                                                                                                                       -1)
            self.best_solution['best_value'] = self.complete_dataset_y[index_subspace_best_value]
            self.hyper_gaussian_object.observations_kernel = []
            self.hyper_gaussian_object.observations_y = []
            self.hyper_gaussian_object.best_kernel = self.best_solution['best_kernel']

            PH.printme(PH.p1, "Best solution is selected as \n", self.best_solution)
            # PH.printme(PH.p1, "\nObservations_kernel:", self.complete_dataset_kernel_observations)
            PH.printme(PH.p1, "Subspace: ", (subspace_selection_count+1), " completed..\n\n\n\n")
            PH.printme(PH.p1, "Data set after:", (subspace_selection_count+1),  " subspace")
            PH.printme(PH.p1, "Observation_y:", self.complete_dataset_y.T, "\n")
            # PH.printme(PH.p1, "Observations_kernel: ", self.complete_dataset_kernel_observations, "\n\n")

        # PH.printme(PH.p1, "plotting results")

        # plotting kernel mean
        if bool_kernel_plot:
            self.hyper_gaussian_object.plot_kernel_wrapper(self.hyper_gaussian_object.current_kernel_samples[0].reshape(1,
                                         self.hyper_gaussian_object.no_principal_components), "Kernel Samples"+str(run_count))
            bool_kernel_plot = False

        # Commenting the following statement to calculate posterior once after few runs
        # self.gp_wrapper_obj.compute_posterior_distribution('HYPER', self.best_solution['best_kernel'], self.hyper_gaussian_object,
        #                                                    "final posterior")

        # Worst solution posterior
        # index_least_value = self.complete_dataset_y.argmin()
        # self.gp_wrapper_obj.compute_posterior_distribution('HYPER', self.complete_dataset_kernel_observations[index_least_value].reshape(
        #     1, -1), self.hyper_gaussian_object,"minimum posterior")

        return self.best_solution




