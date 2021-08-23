import numpy as np
import scipy.optimize as opt
from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

class AIModel:

    def __init__(self, epsilon_distance, minimiser_restarts, lambda_reg, lambda_mul, llk_threshold):
        self.epsilon_distance = epsilon_distance
        self.number_minimiser_restarts = minimiser_restarts
        self.lambda_reg = lambda_reg
        self.lambda_mul = lambda_mul
        self.min_acq_difference = None
        self.max_acq_difference = None
        self.min_llk = None
        self.max_llk = None
        self.llk_threshold = llk_threshold

    def obtain_twostg_aimodel_suggestions(self, plot_files_identifier, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count,
                                   plot_iterations):

        x_max_value = None
        log_like_max = -1 * float("inf")

        best_solutions = np.array([])
        start_points_list = []

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

            maxima = opt.minimize(lambda x: -gp_aimodel.optimize_log_marginal_likelihood_weight_params(x),
                                  tot_init_points,
                                  method='L-BFGS-B',
                                  tol=0.01,
                                  options={'maxfun': 200, 'maxiter': 40},
                                  bounds=total_bounds)

            params = maxima['x']
            log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_weight_params(params)
            if log_likelihood > log_like_max:
                PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for params ", params)
                x_max_value = maxima['x']
                log_like_max = log_likelihood

            start_points_list.append(tot_init_points)
            best_solutions=np.append(best_solutions, np.array(log_likelihood))

        stage_one_best_kernel = x_max_value

        gp_aimodel.len_weights = x_max_value[0:(len(maxima['x']) - 1)]
        gp_aimodel.signal_variance = x_max_value[len(maxima['x']) - 1]

        PH.printme(PH.p1, "******* Stage 1 Optimisation complete *******\nOpt weights: ", gp_aimodel.len_weights,
                   "  Signal variance: ", gp_aimodel.signal_variance, "\n Maximum Liklihood:", log_like_max)

        if gp_aimodel.he_suggestions is not None:

            PH.printme(PH.p1, "\nStarting stage 2")
            # self.constrained_distance_maximiser(gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, stage_one_best_kernel,
            #                                     start_points_list, best_solutions)

            self.distance_maximiser_for_likelihood(gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, stage_one_best_kernel,
                                                start_points_list, best_solutions)

            PH.printme(PH.p1, "******* Stage 2 Optimisation complete *******")

        PH.printme(PH.p1, "Final Optimisation : Opt weights: ", gp_aimodel.len_weights,
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


    def distance_maximiser_for_likelihood(self, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, stage_one_best_kernel,
                                       start_points_list, best_solutions):

        total_bounds = gp_aimodel.len_weights_bounds.copy()
        total_bounds.append(gp_aimodel.signal_variance_bounds)

        # total_bounds = ((0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0))

        compromised_likelihood = self.llk_threshold * best_solutions[0]
        start_point = stage_one_best_kernel

        constraint_list = []
        const_llk = {'type': 'ineq', 'fun': lambda x: gp_aimodel.optimize_log_marginal_likelihood_weight_params(x)[0] -
                                                      compromised_likelihood}

        constraint_list.append(const_llk)

        # # # # # COBYLA
        # cons = {'type': 'ineq', 'fun': lambda x: x}
        # constraint_list.append(cons)
        #
        # for bnd_index in range(len(gp_aimodel.len_weights_bounds)):
        #     lower = gp_aimodel.len_weights_bounds[bnd_index][0]
        #     upper = gp_aimodel.len_weights_bounds[bnd_index][1]
        #     low_cons = {'type': 'ineq', 'fun': lambda x, lb=lower, i=bnd_index: x[i] - lb}
        #     up_cons = {'type': 'ineq', 'fun': lambda x, ub=upper, i=bnd_index: ub - x[i]}
        #     constraint_list.append(low_cons)
        #     constraint_list.append(up_cons)
        #
        # variance_index = len(gp_aimodel.len_weights)
        # const_var_lo = {'type': 'ineq', 'fun': lambda x, lb=gp_aimodel.signal_variance_bounds[0], i=variance_index: x[i] - lb}
        # const_var_up = {'type': 'ineq', 'fun': lambda x, ub=gp_aimodel.signal_variance_bounds[1], i=variance_index: ub - x[i]}
        #
        # constraint_list.append(const_var_lo)
        # constraint_list.append(const_var_up)
        #
        # maxima = opt.minimize(lambda x: -self.constrained_distance_maximiser(x, gp_aimodel, acq_func_obj, noisy_suggestions,
        #                                                                      ai_suggestion_count), start_point, method='COBYLA',
        #                       constraints=constraint_list
        #                       )

        # # # SLSQP
        # maxima = opt.minimize(lambda x: -self.constrained_distance_maximiser(x, gp_aimodel, acq_func_obj, noisy_suggestions,
        #                                                                      ai_suggestion_count), start_point, method='SLSQP',
        #                       constraints=constraint_list, bounds=total_bounds
        #                       )

        # # # SHGO
        maxima = opt.shgo(lambda x: -self.constrained_distance_maximiser(x, gp_aimodel, acq_func_obj,
                                          noisy_suggestions, ai_suggestion_count), total_bounds, constraints=const_llk, n=100, iters=4)

        params = maxima['x']
        distance = self.constrained_distance_maximiser(params, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count)

        PH.printme(PH.p1, "New constrained distance maximum for stage two ", distance, " found for params ", maxima['x'])

        gp_aimodel.len_weights = maxima['x'][0:(len(maxima['x']) - 1)]
        gp_aimodel.signal_variance = maxima['x'][len(maxima['x']) - 1]


    def constrained_distance_maximiser(self, inputs, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count):

        PH.printme(PH.p1, "Distance Maximiser Parameters : ", inputs)

        gp_aimodel.len_weights = inputs[0:(len(inputs) - 1)]
        gp_aimodel.signal_variance = inputs[(len(inputs) - 1)]

        acq_difference_sum = 0

        for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):

            data_conditioned_on_current_he_suggestion_X = gp_aimodel.X[0:(gp_aimodel.number_of_observed_samples +
                                                                          gp_aimodel.HE_input_iterations[i] - 1)]
            data_conditioned_on_current_he_suggestion_y = gp_aimodel.y[0:(gp_aimodel.number_of_observed_samples +
                                                                          gp_aimodel.HE_input_iterations[i] - 1)]

            Xs_random = np.random.uniform(0, 1, 10).reshape(10, 1)
            random_acq_values = []

            if acq_func_obj.acq_type == "ucb":

                best_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
                                                                                           gp_aimodel.he_suggestions[
                                                                                               "x_suggestions_best"][
                                                                                               i],
                                                                                           data_conditioned_on_current_he_suggestion_X,
                                                                                           data_conditioned_on_current_he_suggestion_y,
                                                                                           gp_aimodel,
                                                                                           ai_suggestion_count)

                worst_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
                                                                                            gp_aimodel.he_suggestions[
                                                                                                "x_suggestions_worst"][
                                                                                                i],
                                                                                            data_conditioned_on_current_he_suggestion_X,
                                                                                            data_conditioned_on_current_he_suggestion_y,
                                                                                            gp_aimodel,
                                                                                            ai_suggestion_count)

                for each_Xs in Xs_random:
                    value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions, each_Xs,
                                                                                      data_conditioned_on_current_he_suggestion_X,
                                                                                      data_conditioned_on_current_he_suggestion_y,
                                                                                      gp_aimodel, ai_suggestion_count)
                    random_acq_values.append(value)

            if acq_func_obj.acq_type == "ei":
                y_max = gp_aimodel.y.max()
                best_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
                                                                                         gp_aimodel.he_suggestions[
                                                                                             "x_suggestions_best"][i],
                                                                                         data_conditioned_on_current_he_suggestion_X,
                                                                                         data_conditioned_on_current_he_suggestion_y,
                                                                                         y_max, gp_aimodel)
                worst_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
                                                                                          gp_aimodel.he_suggestions[
                                                                                              "x_suggestions_worst"][i],
                                                                                          data_conditioned_on_current_he_suggestion_X,
                                                                                          data_conditioned_on_current_he_suggestion_y,
                                                                                          y_max, gp_aimodel)

                for each_Xs in Xs_random:
                    value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions, each_Xs,
                                                                                    data_conditioned_on_current_he_suggestion_X,
                                                                                    data_conditioned_on_current_he_suggestion_y,
                                                                                    y_max, gp_aimodel)
                    random_acq_values.append(value)

            best_mean = np.mean(random_acq_values)
            worst_mean = np.mean(worst_acq_value)
            if best_mean != 0:
                best_acq_value = best_acq_value / best_mean
            if worst_mean != 0:
                worst_acq_value = worst_acq_value / worst_mean

            acq_difference_sum += best_acq_value - worst_acq_value

        if acq_difference_sum < self.min_acq_difference:
            self.min_acq_difference = acq_difference_sum
        if acq_difference_sum > self.max_acq_difference:
            self.max_acq_difference = acq_difference_sum

        constrained_likelihood = gp_aimodel.optimize_log_marginal_likelihood_weight_params(inputs)
        PH.printme(PH.p1, "Const. Llk:", constrained_likelihood, "    Dist:", acq_difference_sum, "   Weights:", inputs)

        return acq_difference_sum


    ##############

    # def constrained_distance_maximiser(self, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, stage_one_best_kernel,
    #                                    start_points_list, best_solutions):
    #
    #     x_max_value = None
    #     distance_max = -1 * float("inf")
    #
    #     unflipped_indices = np.argsort(best_solutions)
    #     indices = np.flip(unflipped_indices)
    #     best_solutions = np.vstack(best_solutions)
    #     start_points_list = np.vstack(start_points_list)
    #
    #     count = 0
    #     for index in indices:
    #         if count < 50:
    #
    #             total_bounds = ((0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0))
    #
    #             compromised_likelihood = self.llk_threshold * best_solutions[index]
    #             start_point = start_points_list[index]
    #
    #             constraint_list = []
    #             const_llk = {'type': 'ineq', 'fun': lambda x: gp_aimodel.optimize_log_marginal_likelihood_weight_params(x)[0] -
    #                                                       compromised_likelihood}
    #
    #             constraint_list.append(const_llk)
    #
    #             # # # # COBYLA
    #             # cons = {'type': 'ineq', 'fun': lambda x: x}
    #             # constraint_list.append(cons)
    #
    #             # for bnd_index in range(len(gp_aimodel.len_weights_bounds)):
    #             #     lower = gp_aimodel.len_weights_bounds[bnd_index][0]
    #             #     upper = gp_aimodel.len_weights_bounds[bnd_index][1]
    #             #     low_cons = {'type': 'ineq', 'fun': lambda x, lb=lower, i=bnd_index: x[i] - lb}
    #             #     up_cons = {'type': 'ineq', 'fun': lambda x, ub=upper, i=bnd_index: ub - x[i]}
    #             #     constraint_list.append(low_cons)
    #             #     constraint_list.append(up_cons)
    #             #
    #             # variance_index = len(gp_aimodel.len_weights)
    #             # const_var_lo = {'type': 'ineq', 'fun': lambda x, lb=gp_aimodel.signal_variance_bounds[0], i=variance_index: x[i] - lb}
    #             # const_var_up = {'type': 'ineq', 'fun': lambda x, ub=gp_aimodel.signal_variance_bounds[1], i=variance_index: ub - x[i]}
    #             #
    #             # constraint_list.append(const_var_lo)
    #             # constraint_list.append(const_var_up)
    #
    #             maxima = opt.minimize(lambda x: -self.distance_maximiser(x, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count),
    #                                   start_point,
    #                                   method='SLSQP', constraints=constraint_list, bounds=total_bounds
    #                                   )
    #
    #             params = maxima['x']
    #             distance = self.distance_maximiser(params, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count)
    #             if distance > distance_max:
    #                 PH.printme(PH.p1, "New constrained distance maximum for stage two ", distance, " found for params ", params)
    #                 distance_max = distance
    #                 x_max_value = maxima['x']
    #
    #     gp_aimodel.len_weights = x_max_value[0:(len(maxima['x']) - 1)]
    #     gp_aimodel.signal_variance = x_max_value[len(maxima['x']) - 1]
    #
    # def distance_maximiser(self, inputs, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count):
    #
    #     gp_aimodel.len_weights = inputs[0:(len(inputs) - 1)]
    #     gp_aimodel.signal_variance = inputs[(len(inputs) - 1)]
    #
    #     acq_difference_sum = 0
    #
    #     for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):
    #
    #         data_conditioned_on_current_he_suggestion_X = gp_aimodel.X[0:(gp_aimodel.number_of_observed_samples +
    #                                                                       gp_aimodel.HE_input_iterations[i] - 1)]
    #         data_conditioned_on_current_he_suggestion_y = gp_aimodel.y[0:(gp_aimodel.number_of_observed_samples +
    #                                                                       gp_aimodel.HE_input_iterations[i] - 1)]
    #
    #         Xs_random = np.random.uniform(0, 1, 10).reshape(10, 1)
    #         random_acq_values = []
    #
    #         if acq_func_obj.acq_type == "ucb":
    #
    #             best_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
    #                                                                                        gp_aimodel.he_suggestions[
    #                                                                                            "x_suggestions_best"][
    #                                                                                            i],
    #                                                                                        data_conditioned_on_current_he_suggestion_X,
    #                                                                                        data_conditioned_on_current_he_suggestion_y,
    #                                                                                        gp_aimodel,
    #                                                                                        ai_suggestion_count)
    #
    #             worst_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
    #                                                                                         gp_aimodel.he_suggestions[
    #                                                                                             "x_suggestions_worst"][
    #                                                                                             i],
    #                                                                                         data_conditioned_on_current_he_suggestion_X,
    #                                                                                         data_conditioned_on_current_he_suggestion_y,
    #                                                                                         gp_aimodel,
    #                                                                                         ai_suggestion_count)
    #
    #             for each_Xs in Xs_random:
    #                 value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions, each_Xs,
    #                                                                                   data_conditioned_on_current_he_suggestion_X,
    #                                                                                   data_conditioned_on_current_he_suggestion_y,
    #                                                                                   gp_aimodel, ai_suggestion_count)
    #                 random_acq_values.append(value)
    #
    #         if acq_func_obj.acq_type == "ei":
    #             y_max = gp_aimodel.y.max()
    #             best_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
    #                                                                                      gp_aimodel.he_suggestions[
    #                                                                                          "x_suggestions_best"][i],
    #                                                                                      data_conditioned_on_current_he_suggestion_X,
    #                                                                                      data_conditioned_on_current_he_suggestion_y,
    #                                                                                      y_max, gp_aimodel)
    #             worst_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
    #                                                                                       gp_aimodel.he_suggestions[
    #                                                                                           "x_suggestions_worst"][i],
    #                                                                                       data_conditioned_on_current_he_suggestion_X,
    #                                                                                       data_conditioned_on_current_he_suggestion_y,
    #                                                                                       y_max, gp_aimodel)
    #
    #             for each_Xs in Xs_random:
    #                 value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions, each_Xs,
    #                                                                                 data_conditioned_on_current_he_suggestion_X,
    #                                                                                 data_conditioned_on_current_he_suggestion_y,
    #                                                                                 y_max, gp_aimodel)
    #                 random_acq_values.append(value)
    #
    #         if best_acq_value != 0:
    #             best_acq_value = best_acq_value / np.mean(random_acq_values)
    #         if worst_acq_value != 0:
    #             worst_acq_value = worst_acq_value / np.mean(worst_acq_value)
    #
    #         acq_difference_sum += best_acq_value - worst_acq_value
    #
    #     if acq_difference_sum < self.min_acq_difference:
    #         self.min_acq_difference = acq_difference_sum
    #     if acq_difference_sum > self.max_acq_difference:
    #         self.max_acq_difference = acq_difference_sum
    #
    #     return acq_difference_sum
    #
    #
    #
    # ############################################
    #
    # def obtain_aimodel_suggestions(self, plot_files_identifier, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count,
    #                                plot_iterations):
    #
    #     x_max_value = None
    #     log_like_distance_max = -1 * float("inf")
    #
    #     random_points_a = []
    #     random_points_b = []
    #     random_points_c = []
    #     random_points_d = []
    #     random_points_e = []
    #
    #     # Data structure to create the starting points for the scipy.minimize method
    #     random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[0][0], gp_aimodel.len_weights_bounds[0][1],
    #                                                    self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
    #     random_points_a.append(random_data_point_each_dim)
    #
    #     random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[1][0], gp_aimodel.len_weights_bounds[1][1],
    #                                                    self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
    #     random_points_b.append(random_data_point_each_dim)
    #
    #     random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[2][0], gp_aimodel.len_weights_bounds[2][1],
    #                                                    self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
    #     random_points_c.append(random_data_point_each_dim)
    #
    #     random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[3][0], gp_aimodel.len_weights_bounds[3][1],
    #                                                    self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
    #     random_points_d.append(random_data_point_each_dim)
    #
    #     random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[4][0], gp_aimodel.len_weights_bounds[4][1],
    #                                                    self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
    #     random_points_e.append(random_data_point_each_dim)
    #
    #     variance_start_points = np.random.uniform(gp_aimodel.signal_variance_bounds[0], gp_aimodel.signal_variance_bounds[1],
    #                                               self.number_minimiser_restarts)
    #
    #     # Vertically stack the arrays of randomly generated starting points as a matrix
    #     random_points_a = np.vstack(random_points_a)
    #     random_points_b = np.vstack(random_points_b)
    #     random_points_c = np.vstack(random_points_c)
    #     random_points_d = np.vstack(random_points_d)
    #     random_points_e = np.vstack(random_points_e)
    #
    #     for ind in np.arange(self.number_minimiser_restarts):
    #
    #         tot_init_points = []
    #
    #         param_a = random_points_a[0][ind]
    #         tot_init_points.append(param_a)
    #         param_b = random_points_b[0][ind]
    #         tot_init_points.append(param_b)
    #         param_c = random_points_c[0][ind]
    #         tot_init_points.append(param_c)
    #         param_d = random_points_d[0][ind]
    #         tot_init_points.append(param_d)
    #         param_e = random_points_e[0][ind]
    #         tot_init_points.append(param_e)
    #         tot_init_points.append(variance_start_points[ind])
    #         total_bounds = gp_aimodel.len_weights_bounds.copy()
    #         # total_bounds.append(gp_aimodel.bounds)
    #         total_bounds.append(gp_aimodel.signal_variance_bounds)
    #
    #         maxima = opt.minimize(lambda x: -self.likelihood_distance_maximiser(x, gp_aimodel, acq_func_obj, noisy_suggestions,
    #                                                                             ai_suggestion_count),
    #                               tot_init_points,
    #                               method='L-BFGS-B',
    #                               tol=0.01,
    #                               options={'maxfun': 200, 'maxiter': 40},
    #                               bounds=total_bounds)
    #
    #         params = maxima['x']
    #         log_likelihood_distance = self.likelihood_distance_maximiser(params, gp_aimodel, acq_func_obj, noisy_suggestions,
    #                                                                      ai_suggestion_count)
    #         if log_likelihood_distance > log_like_distance_max:
    #             PH.printme(PH.p1, "New maximum log likelihood distance:", log_likelihood_distance, " found for params ", params)
    #             x_max_value = maxima['x']
    #             log_like_distance_max = log_likelihood_distance
    #
    #     gp_aimodel.len_weights = x_max_value[0:(len(maxima['x']) - 1)]
    #     gp_aimodel.signal_variance = x_max_value[len(maxima['x']) - 1]
    #
    #     PH.printme(PH.p1, "*******After minimising distance \t Observing the values *******\nOpt weights: ", gp_aimodel.len_weights,
    #                "  Signal variance: ", gp_aimodel.signal_variance)
    #
    #     xnew, acq_func_values = acq_func_obj.max_acq_func("ai", noisy_suggestions, gp_aimodel, ai_suggestion_count)
    #
    #     # uncomment to  plot Acq functions
    #     if gp_aimodel.number_of_dimensions == 1 and plot_iterations != 0 and ai_suggestion_count % plot_iterations == 0:
    #         plot_axes = [0, 1, acq_func_values.min() * 0.7, acq_func_values.max() * 2]
    #         # print(acq_func_values)
    #         acq_func_obj.plot_acquisition_function(plot_files_identifier + "acq_" + str(ai_suggestion_count), gp_aimodel.Xs,
    #                                                acq_func_values, plot_axes)
    #
    #     PH.printme(PH.p1, "Best value for acq function is found at ", xnew)
    #
    #     return xnew
    #
    # def likelihood_distance_maximiser(self, inputs, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count):
    #
    #     lambda_val = self.lambda_reg * self.lambda_mul
    #
    #     log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_weight_params(inputs)
    #
    #     if gp_aimodel.he_suggestions is None:
    #         return log_likelihood
    #
    #     gp_aimodel.len_weights = inputs[0:(len(inputs) - 1)]
    #     gp_aimodel.signal_variance = inputs[(len(inputs) - 1)]
    #
    #     acq_difference_sum = 0
    #
    #     for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):
    #
    #         data_conditioned_on_current_he_suggestion_X = gp_aimodel.X[0:(gp_aimodel.number_of_observed_samples +
    #                                                                        gp_aimodel.HE_input_iterations[i] - 1)]
    #         data_conditioned_on_current_he_suggestion_y = gp_aimodel.y[0:(gp_aimodel.number_of_observed_samples +
    #                                                                       gp_aimodel.HE_input_iterations[i] - 1)]
    #
    #         Xs_random = np.random.uniform(0, 1, 10).reshape(10, 1)
    #         random_acq_values = []
    #
    #
    #         if acq_func_obj.acq_type == "ucb":
    #
    #             best_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
    #                                                                                        gp_aimodel.he_suggestions["x_suggestions_best"][
    #                                                                                            i],
    #                                                                                        data_conditioned_on_current_he_suggestion_X,
    #                                                                                        data_conditioned_on_current_he_suggestion_y,
    #                                                                                        gp_aimodel,
    #                                                                                        ai_suggestion_count)
    #
    #             worst_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
    #                                                                                         gp_aimodel.he_suggestions[
    #                                                                                             "x_suggestions_worst"][i],
    #                                                                                         data_conditioned_on_current_he_suggestion_X,
    #                                                                                         data_conditioned_on_current_he_suggestion_y,
    #                                                                                         gp_aimodel, ai_suggestion_count)
    #
    #             for each_Xs in Xs_random:
    #                 value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions, each_Xs,
    #                                                                                           data_conditioned_on_current_he_suggestion_X,
    #                                                                                           data_conditioned_on_current_he_suggestion_y,
    #                                                                                           gp_aimodel, ai_suggestion_count)
    #                 random_acq_values.append(value)
    #
    #
    #         if acq_func_obj.acq_type == "ei":
    #             y_max = gp_aimodel.y.max()
    #             best_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
    #                                                                     gp_aimodel.he_suggestions["x_suggestions_best"][i],
    #                                                                                      data_conditioned_on_current_he_suggestion_X,
    #                                                                                      data_conditioned_on_current_he_suggestion_y,
    #                                                                                      y_max, gp_aimodel)
    #             worst_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
    #                                                                     gp_aimodel.he_suggestions["x_suggestions_worst"][i],
    #                                                                                      data_conditioned_on_current_he_suggestion_X,
    #                                                                                      data_conditioned_on_current_he_suggestion_y,
    #                                                                                       y_max, gp_aimodel)
    #
    #             for each_Xs in Xs_random:
    #                 value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions, each_Xs,
    #                                                                                      data_conditioned_on_current_he_suggestion_X,
    #                                                                                      data_conditioned_on_current_he_suggestion_y,
    #                                                                                       y_max, gp_aimodel)
    #                 random_acq_values.append(value)
    #
    #         if best_acq_value != 0:
    #             best_acq_value = best_acq_value / np.mean(random_acq_values)
    #         if worst_acq_value != 0:
    #             worst_acq_value = worst_acq_value/np.mean(worst_acq_value)
    #
    #         acq_difference_sum += best_acq_value - worst_acq_value
    #
    #     value = log_likelihood + lambda_val * acq_difference_sum
    #
    #     if acq_difference_sum < self.min_acq_difference:
    #         self.min_acq_difference = acq_difference_sum
    #     if acq_difference_sum > self.max_acq_difference:
    #         self.max_acq_difference = acq_difference_sum
    #
    #     if log_likelihood < self.min_llk:
    #         self.min_llk = log_likelihood
    #     if log_likelihood > self.max_llk:
    #         self.max_llk = log_likelihood
    #
    #     return value



