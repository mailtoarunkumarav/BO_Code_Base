import numpy as np

from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

class HumanExpertModel:

    def initiate_humanexpert_model(self, start_time, run_count, gp_groundtruth, gp_wrapper_obj, acq_func_obj,
                                   number_of_observations_humanexpert, number_of_humanexpert_suggestions):

        PH.printme(PH.p1, "Construct GP object for Expert")
        gp_humanexpert = gp_wrapper_obj.construct_gp_object(start_time, "HumanExpert", number_of_observations_humanexpert, None)

        initial_random_expert_observations_X = gp_humanexpert.X
        initial_random_expert_observations_y = gp_humanexpert.y

        gp_humanexpert.initial_random_observations = {"observations_X": initial_random_expert_observations_X, "observations_y":
            initial_random_expert_observations_y}

        # Ground truth knowledge available for Expert
        gp_humanexpert.len_weights = gp_groundtruth.len_weights
        gp_humanexpert.signal_variance = gp_groundtruth.signal_variance

        gp_humanexpert.runGaussian("R" + str(run_count) + "_" + start_time, "HE_Initial")

        # Obtain Suggestions
        suggestions_X, suggestions_y = self.obtain_expert_suggestions(run_count, start_time, gp_humanexpert, acq_func_obj,
                                                                      number_of_humanexpert_suggestions)

        PH.printme(PH.p1, number_of_humanexpert_suggestions, "Expert suggestions:\n", suggestions_X.T, "\n", suggestions_y.T)
        gp_humanexpert.runGaussian("R" + str(run_count) + "_" + start_time, "_HE_final")

        gp_humanexpert.suggestions = {"suggestions_X": suggestions_X, "suggestions_y": suggestions_y}

        return gp_humanexpert

    def obtain_expert_suggestions(self, run_count, start_time, gp_humanexpert, acq_func_obj, number_of_humanexpert_suggestions):

        suggestions_X = []
        suggestions_y = []
        PH.printme(PH.p1, "Initial X and y:\n", gp_humanexpert.X, "\n", gp_humanexpert.y)

        for suggestion_count in range(number_of_humanexpert_suggestions):
            PH.printme(PH.p1, "Compute Suggestion: ", suggestion_count + 1)
            xnew, acq_func_values = acq_func_obj.max_acq_func(gp_humanexpert, suggestion_count + 1)
            xnew_orig = np.multiply(xnew.T, (gp_humanexpert.Xmax - gp_humanexpert.Xmin)) + gp_humanexpert.Xmin

            # Add the new observation point to the existing set of observed samples along with its true value
            X = gp_humanexpert.X
            X = np.append(X, [xnew], axis=0)

            ynew_orig = gp_humanexpert.fun_helper_obj.get_true_func_value(xnew_orig)
            ynew = (ynew_orig - gp_humanexpert.ymin) / (gp_humanexpert.ymax - gp_humanexpert.ymin)

            y = np.append(gp_humanexpert.y, [ynew], axis=0)

            # Normalising
            PH.printme(PH.p1, "(", xnew, ynew, ") is the new value added..    Original: ", (xnew_orig, ynew_orig))
            PH.printme(PH.p1, "Suggestion: ", suggestion_count + 1, " complete")
            gp_humanexpert.X = X
            gp_humanexpert.y = y

            suggestions_X.append(xnew)
            suggestions_y.append(ynew)

            # # Uncomment for debugging the suggestions and the posterior
            # PH.printme(PH.p1, "Final X and y:\n", gp_humanexpert.X, "\n", gp_humanexpert.y)

            with np.errstate(invalid='ignore'):
                mean, diag_variance, f_prior, f_post = gp_humanexpert.gaussian_predict(gp_humanexpert.Xs)
                standard_deviation = np.sqrt(diag_variance)
            gp_humanexpert.plot_posterior_predictions("R" + str(run_count) + "_" + start_time+"_HE_Suggestion" + "_" + str(
                suggestion_count+1), gp_humanexpert.Xs,  gp_humanexpert.ys, mean, standard_deviation)

        suggestions_X = np.vstack(suggestions_X)
        suggestions_y = np.vstack(suggestions_y)
        return suggestions_X, suggestions_y
