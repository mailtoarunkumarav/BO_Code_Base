import numpy as np
import scipy.optimize as opt
from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

class AIModel:

    def __init__(self, epsilon_distance, minimiser_restarts, lambda_reg, lambda_mul):
        self.epsilon_distance = epsilon_distance
        self.number_minimiser_restarts = minimiser_restarts
        self.lambda_reg = lambda_reg
        self.lambda_mul = lambda_mul
        self.min_acq_difference = None
        self.max_acq_difference = None
        self.min_llk = None
        self.max_llk = None

    def obtain_aimodel_suggestions(self, plot_files_identifier, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count,
                                   plot_iterations):

        x_max_value = None
        log_like_distance_max = -1 * float("inf")

        random_points_a = []
        random_points_b = []
        random_points_c = []
        random_points_d = []
        random_points_e = []

        # Data structure to create the starting points for the scipy.minimize method
        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[0][0], gp_aimodel.len_weights_bounds[0][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_a.append(random_data_point_each_dim)

        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[1][0], gp_aimodel.len_weights_bounds[1][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_b.append(random_data_point_each_dim)

        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[2][0], gp_aimodel.len_weights_bounds[2][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_c.append(random_data_point_each_dim)

        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[3][0], gp_aimodel.len_weights_bounds[3][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_d.append(random_data_point_each_dim)

        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[4][0], gp_aimodel.len_weights_bounds[4][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_e.append(random_data_point_each_dim)

        variance_start_points = np.random.uniform(gp_aimodel.signal_variance_bounds[0], gp_aimodel.signal_variance_bounds[1],
                                                  self.number_minimiser_restarts)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points_a = np.vstack(random_points_a)
        random_points_b = np.vstack(random_points_b)
        random_points_c = np.vstack(random_points_c)
        random_points_d = np.vstack(random_points_d)
        random_points_e = np.vstack(random_points_e)

        for ind in np.arange(self.number_minimiser_restarts):

            tot_init_points = []

            param_a = random_points_a[0][ind]
            tot_init_points.append(param_a)
            param_b = random_points_b[0][ind]
            tot_init_points.append(param_b)
            param_c = random_points_c[0][ind]
            tot_init_points.append(param_c)
            param_d = random_points_d[0][ind]
            tot_init_points.append(param_d)
            param_e = random_points_e[0][ind]
            tot_init_points.append(param_e)
            tot_init_points.append(variance_start_points[ind])
            total_bounds = gp_aimodel.len_weights_bounds.copy()
            # total_bounds.append(gp_aimodel.bounds)
            total_bounds.append(gp_aimodel.signal_variance_bounds)

            maxima = opt.minimize(lambda x: -self.likelihood_distance_maximiser(x, gp_aimodel, acq_func_obj, noisy_suggestions,
                                                                                ai_suggestion_count),
                                  tot_init_points,
                                  method='L-BFGS-B',
                                  tol=0.01,
                                  options={'maxfun': 200, 'maxiter': 40},
                                  bounds=total_bounds)

            params = maxima['x']
            log_likelihood_distance = self.likelihood_distance_maximiser(params, gp_aimodel, acq_func_obj, noisy_suggestions,
                                                                         ai_suggestion_count)
            if log_likelihood_distance > log_like_distance_max:
                PH.printme(PH.p1, "New maximum log likelihood distance:", log_likelihood_distance, " found for params ", params)
                x_max_value = maxima['x']
                log_like_distance_max = log_likelihood_distance

        gp_aimodel.len_weights = x_max_value[0:(len(maxima['x']) - 1)]
        gp_aimodel.signal_variance = x_max_value[len(maxima['x']) - 1]

        PH.printme(PH.p1, "*******After minimising distance \t Observing the values *******\nOpt weights: ", gp_aimodel.len_weights,
                   "  Signal variance: ", gp_aimodel.signal_variance)

        xnew, acq_func_values = acq_func_obj.max_acq_func("ai", noisy_suggestions, gp_aimodel, ai_suggestion_count)

        # uncomment to  plot Acq functions
        if gp_aimodel.number_of_dimensions == 1 and plot_iterations != 0 and ai_suggestion_count % plot_iterations == 0:
            plot_axes = [0, 1, acq_func_values.min() * 0.7, acq_func_values.max() * 2]
            # print(acq_func_values)
            acq_func_obj.plot_acquisition_function(plot_files_identifier + "acq_" + str(ai_suggestion_count), gp_aimodel.Xs,
                                                   acq_func_values, plot_axes)

        PH.printme(PH.p1, "Best value for acq function is found at ", xnew)

        return xnew

    def likelihood_distance_maximiser(self, inputs, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count):

        lambda_val = self.lambda_reg * self.lambda_mul

        log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_weight_params(inputs)

        if gp_aimodel.he_suggestions is None:
            return log_likelihood

        gp_aimodel.len_weights = inputs[0:(len(inputs) - 1)]
        gp_aimodel.signal_variance = inputs[(len(inputs) - 1)]

        acq_difference_sum = 0

        if acq_func_obj.acq_type == "ucb":

            for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):
                best_acq_value = acq_func_obj.upper_confidence_bound_util("ai", noisy_suggestions,
                                                                          gp_aimodel.he_suggestions["x_suggestions_best"][i],
                                                                          gp_aimodel,
                                                                          ai_suggestion_count)
                worst_acq_value = acq_func_obj.upper_confidence_bound_util("ai", noisy_suggestions,
                                                                           gp_aimodel.he_suggestions["x_suggestions_worst"][i],
                                                                           gp_aimodel, ai_suggestion_count)
                acq_difference_sum += best_acq_value - worst_acq_value

        if acq_func_obj.acq_type == "ei":
            y_max = gp_aimodel.y.max()
            for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):
                best_acq_value = acq_func_obj.expected_improvement_util("ai", noisy_suggestions,
                                                                        gp_aimodel.he_suggestions["x_suggestions_best"][i], y_max,
                                                                        gp_aimodel)
                worst_acq_value = acq_func_obj.expected_improvement_util("ai", noisy_suggestions, gp_aimodel.he_suggestions[
                    "x_suggestions_worst"][i], y_max, gp_aimodel)
                acq_difference_sum += best_acq_value - worst_acq_value

        value = log_likelihood + lambda_val * acq_difference_sum

        if acq_difference_sum < self.min_acq_difference:
            self.min_acq_difference = acq_difference_sum
        if acq_difference_sum > self.max_acq_difference:
            self.max_acq_difference = acq_difference_sum

        if log_likelihood < self.min_llk:
            self.min_llk = log_likelihood
        if log_likelihood > self.max_llk:
            self.max_llk = log_likelihood

        return value



