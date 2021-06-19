import numpy as np
import scipy.optimize as opt
from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

class AIModel:

    def __init__(self, epsilon_distance, minimiser_restarts):
        self.epsilon_distance = epsilon_distance
        self.number_minimiser_restarts = minimiser_restarts


    def initiate_aimodel(self, start_time, run_count, gp_wrapper_obj, gp_humanexpert, acq_func_obj, number_of_observations_humanexpert,
                         number_of_ai_suggestions):

        PH.printme(PH.p1, "Initialising AI model for kernel optimisation")

        ai_model_observations_X = gp_humanexpert.initial_random_observations["observations_X"]
        ai_model_observations_y = gp_humanexpert.initial_random_observations["observations_y"]
        gp_aimodel = gp_wrapper_obj.construct_gp_object(start_time, "ai", number_of_observations_humanexpert,
                                                        gp_humanexpert.initial_random_observations)
        gp_aimodel.he_suggestions = gp_humanexpert.suggestions
        self.optimise_kernel(gp_aimodel, run_count, start_time, number_of_ai_suggestions, acq_func_obj)
        return gp_aimodel

    def optimise_kernel(self, gp_aimodel, run_count, start_time, number_of_ai_suggestions, acq_func_obj):
        PH.printme(PH.p1, "Optimising the kernel with the suggested observations")

        ai_suggestions_X = []
        ai_suggestions_y = []

        for ai_suggestion_count in range(number_of_ai_suggestions):

            # Modified minimiser for the updated kernel estimation
            PH.printme(PH.p1, "Flipped constraints on distance minimisation......")
            PH.printme(PH.p1, "AI model is predicting in iteration: ", ai_suggestion_count+1)
            xnew_suggestion = self.minimise_suggestions_distance(gp_aimodel, acq_func_obj, ai_suggestion_count)
            # print(xnew_suggestion)
            # exit()
            xnew_orig = np.multiply(xnew_suggestion.T, (gp_aimodel.Xmax - gp_aimodel.Xmin)) + gp_aimodel.Xmin

            # Add the new observation point to the existing set of observed samples along with its true value
            X = gp_aimodel.X
            X = np.append(X, [xnew_suggestion], axis=0)

            ynew_orig = gp_aimodel.fun_helper_obj.get_true_func_value(xnew_orig)
            ynew_suggestion= (ynew_orig - gp_aimodel.ymin) / (gp_aimodel.ymax - gp_aimodel.ymin)

            y = np.append(gp_aimodel.y, [ynew_suggestion], axis=0)

            # Normalising
            PH.printme(PH.p1, "(", xnew_suggestion, ynew_suggestion, ") is the new value added..    Original: ", (xnew_orig, ynew_orig))
            PH.printme(PH.p1, "Suggestion: ", ai_suggestion_count + 1, " complete")
            gp_aimodel.X = X
            gp_aimodel.y = y

            ai_suggestions_X.append(xnew_suggestion)
            ai_suggestions_y.append(ynew_suggestion)

            # # Uncomment for debugging the suggestions and the posterior
            PH.printme(PH.p1, "Final X and y:\n", gp_aimodel.X, "\n", gp_aimodel.y)

            with np.errstate(invalid='ignore'):
                mean, diag_variance, f_prior, f_post = gp_aimodel.gaussian_predict(gp_aimodel.Xs)
                standard_deviation = np.sqrt(diag_variance)
            gp_aimodel.plot_posterior_predictions("R" + str(run_count+1) + "_" + start_time + "_ai_suggestion" + "_" + str(
                ai_suggestion_count+1), gp_aimodel.Xs, gp_aimodel.ys, mean, standard_deviation)

    def minimise_suggestions_distance(self, gp_aimodel, acq_func_obj, ai_suggestion_count):

        x_min_value = None
        distance_max = 1 * float("inf")

        random_points_a = []
        random_points_b = []
        random_points_c = []
        # random_points_d = []

        # Data structure to create the starting points for the scipy.minimize method
        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[0][0],
                                                       gp_aimodel.len_weights_bounds[0][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_a.append(random_data_point_each_dim)

        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[1][0],
                                                       gp_aimodel.len_weights_bounds[1][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_b.append(random_data_point_each_dim)

        random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[2][0],
                                                       gp_aimodel.len_weights_bounds[2][1],
                                                       self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        random_points_c.append(random_data_point_each_dim)

        # random_data_point_each_dim = np.random.uniform(self.bounds[0][0],
        #                                                self.bounds[0][1],
        #                                                self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
        # random_points_d.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points_a = np.vstack(random_points_a)
        random_points_b = np.vstack(random_points_b)
        random_points_c = np.vstack(random_points_c)
        # random_points_d = np.vstack(random_points_d)

        for ind in np.arange(self.number_minimiser_restarts):

            tot_init_points = []

            param_a = random_points_a[0][ind]
            tot_init_points.append(param_a)
            param_b = random_points_b[0][ind]
            tot_init_points.append(param_b)
            param_c = random_points_c[0][ind]
            tot_init_points.append(param_c)
            # param_d = random_points_d[0][ind]
            # tot_init_points.append(param_d)
            total_bounds = gp_aimodel.len_weights_bounds.copy()
            # total_bounds.append(gp_aimodel.bounds)

            minima = opt.minimize(lambda x: self.distance_minimiser(x, gp_aimodel, acq_func_obj, ai_suggestion_count),
                                  tot_init_points,
                                  method='L-BFGS-B',
                                  tol=0.01,
                                  options={'maxfun': 20, 'maxiter': 20},
                                  bounds=total_bounds)

            params = minima['x']
            min_distance = self.distance_minimiser(params, gp_aimodel, acq_func_obj, ai_suggestion_count, PH.p1)
            if min_distance < distance_max:
                PH.printme(PH.p1, "New minimum distance found: ", min_distance, " found for params ", params)
                PH.printme(PH.p1, "###############This is the buffered xnew", self.xnew_temp, "#######################")
                x_min_value = minima['x']
                distance_max = min_distance

        gp_aimodel.len_weights = x_min_value[0:3]
        # suggestion_Xnew = x_min_value[len(minima['x']) - 1]

        PH.printme(PH.p1, "Opt weights: ", gp_aimodel.len_weights,
                   # "   Suggestion_Xnew:", self.signal_variance
                   )
        xnew_suggestion, acq_func_values = acq_func_obj.max_acq_func(gp_aimodel, ai_suggestion_count + 1)
        PH.printme(PH.p1, "xnew_suggestion: ", xnew_suggestion)
        return xnew_suggestion

    def distance_minimiser(self, inputs, gp_aimodel, acq_func_obj, ai_suggestion_count, print_bool="FF"):

        gp_aimodel.len_weights = inputs[0:3]
        # Following parameters not used in any computations

        xnew, acq_func_values = acq_func_obj.max_acq_func(gp_aimodel, ai_suggestion_count + 1, print_bool)
        self.xnew_temp = xnew
        PH.printme(print_bool, "Best value for acq function is found at ", xnew)
        distance_between_ind_suggestions = np.linalg.norm(gp_aimodel.he_suggestions["suggestions_X"][-1]-xnew)

        if distance_between_ind_suggestions > self.epsilon_distance:
            total_minimiser_value = 1 * float("inf")
        else:
            K_x_x = gp_aimodel.multi_kernel(gp_aimodel.X, gp_aimodel.X, gp_aimodel.char_len_scale, gp_aimodel.signal_variance)
            eye = 1e-3 * np.eye(len(gp_aimodel.X))
            Knoise = K_x_x + eye
            # Find L from K = L *L.T instead of inversing the covariance function
            L_x_x = np.linalg.cholesky(Knoise)
            factor = np.linalg.solve(L_x_x, gp_aimodel.y)

            # multiplied with -1 to maximise the likelihood
            log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) + gp_aimodel.number_of_observed_samples * np.log(2 * np.pi) +
                                              np.log(np.linalg.det(Knoise)))
            total_minimiser_value = -1 * log_marginal_likelihood + distance_between_ind_suggestions

        return total_minimiser_value







