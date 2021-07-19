import numpy as np

from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

class HumanExpertModel:

    def construct_human_expert_model(self, run_count, pwd_qualifier, number_of_observations_groundtruth, function_type, gp_wrapper_obj,
                                     number_of_random_observations_humanexpert, noisy_suggestions):

        PH.printme(PH.p1, "Constructing Kernel for ground truth....")
        gp_groundtruth = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "GroundTruth", number_of_observations_groundtruth,
                                                            function_type, None)
        gp_groundtruth.runGaussian(pwd_qualifier + "R" + str(run_count + 1), "GT")
        PH.printme(PH.p1, "Ground truth kernel construction complete")

        PH.printme(PH.p1, "Construct GP object for Expert")
        gp_humanexpert = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "HumanExpert",
                                                            number_of_random_observations_humanexpert, function_type, None)
        gp_humanexpert.initial_random_observations = {"observations_X": gp_humanexpert.X, "observations_y": gp_humanexpert.y}

        PH.printme(PH.p1, "Human Expert with info from Ground truth kernel: ", gp_groundtruth.len_weights)

        if noisy_suggestions:
            PH.printme(PH.p1, "Adding noise to the Human expert model....")
            gp_humanexpert.len_weights = np.array([])
            for i in range(len(gp_groundtruth.len_weights)):
                if gp_groundtruth.len_weights[i] == 0:
                    value = gp_groundtruth.len_weights[i] + 0.1
                elif gp_groundtruth.len_weights[i] == 1:
                    value = gp_groundtruth.len_weights[i] - 0.1
                else:
                    value = np.random.normal(gp_groundtruth.len_weights[i], 0.05)
                gp_humanexpert.len_weights = np.append(gp_humanexpert.len_weights, value)
            PH.printme(PH.p1, "After adding noise to the Human Expert model", gp_humanexpert.len_weights)

        else:
            gp_humanexpert.len_weights = gp_groundtruth.len_weights

        gp_humanexpert.signal_variance = gp_groundtruth.signal_variance

        gp_humanexpert.runGaussian(pwd_qualifier + "R" + str(run_count + 1), "HE_Initial")
        return gp_humanexpert

    def obtain_human_expert_suggestions(self, suggestion_count, file_identifier, gp_humanexpert, acq_func_obj, noisy_suggestions,
                                        plot_iterations):

        PH.printme(PH.p1, "Compute Suggestion: ", suggestion_count)
        xnew_best, acq_func_values_best = acq_func_obj.max_acq_func("HumanExpert", noisy_suggestions, gp_humanexpert, suggestion_count)
        xnew_orig_best = np.multiply(xnew_best.T, (gp_humanexpert.Xmax - gp_humanexpert.Xmin)) + gp_humanexpert.Xmin

        ynew_orig_best = gp_humanexpert.fun_helper_obj.get_true_func_value(xnew_orig_best)
        ynew_best = (ynew_orig_best - gp_humanexpert.ymin) / (gp_humanexpert.ymax - gp_humanexpert.ymin)

        # plot_axes = [0, 1, acq_func_values_best.min() * 0.7, acq_func_values_best.max() * 2]
        # acq_func_obj.plot_acquisition_function(file_identifier + "Test_HE_Best_acq_" + str(suggestion_count), gp_humanexpert.Xs,
        #                                        acq_func_values_best, plot_axes)

        # Normalising
        PH.printme(PH.p1, "(", xnew_best, ynew_best, ") is the new best value added..    Original: ",
                   (xnew_orig_best, ynew_orig_best))

        xnew_worst, acq_func_values_worst = acq_func_obj.min_acq_func("HumanExpert", noisy_suggestions, gp_humanexpert, suggestion_count)
        xnew_orig_worst = np.multiply(xnew_worst.T, (gp_humanexpert.Xmax - gp_humanexpert.Xmin)) + gp_humanexpert.Xmin

        # plot_axes = [0, 1, acq_func_values_worst.min() * 0.7, acq_func_values_worst.max() * 2]
        # acq_func_obj.plot_acquisition_function(file_identifier + "_Test_HE_acq_" + str(suggestion_count), gp_humanexpert.Xs,
        #                                        acq_func_values_worst, plot_axes)

        PH.printme(PH.p1, "This is the worst value added after minimising Acq. ", "\tXworst: ", xnew_worst, "\tOriginal:", xnew_orig_worst)

        # Add the new observation point to the existing set of observed samples along with its true value
        X = gp_humanexpert.X
        X = np.append(X, [xnew_best], axis=0)

        y = np.append(gp_humanexpert.y, [ynew_best], axis=0)

        gp_humanexpert.X = X
        gp_humanexpert.y = y

        # Plotting the posteriors
        if gp_humanexpert.number_of_dimensions == 1 and plot_iterations != 0 and suggestion_count % plot_iterations == 0:
            with np.errstate(invalid='ignore'):
                mean, diag_variance, f_prior, f_post = gp_humanexpert.gaussian_predict(gp_humanexpert.Xs)
                standard_deviation = np.sqrt(diag_variance)
            gp_humanexpert.plot_posterior_predictions(file_identifier + "_Suggestion" + "_" + str(
                suggestion_count), gp_humanexpert.Xs, gp_humanexpert.ys, mean, standard_deviation)

        return xnew_best, xnew_worst
