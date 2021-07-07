import numpy as np

from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

class BaselineModel:

    def initiate_baseline_model(self, start_time, run_count, gp_wrapper_obj, gp_humanexpert, acq_func_obj,
                                   number_of_random_observations_humanexpert, number_of_baseline_suggestions):

        PH.printme(PH.p1, "Construct GP object for baseline")
        gp_baseline = gp_wrapper_obj.construct_gp_object(start_time, "baseline",
                                                         number_of_random_observations_humanexpert,
                                                         gp_humanexpert.initial_random_observations)

        initial_random_baseline_observations_X = gp_baseline.X
        initial_random_baseline_observations_y = gp_baseline.y

        gp_baseline.initial_random_observations = {"observations_X": initial_random_baseline_observations_X, "observations_y":
            initial_random_baseline_observations_y}

        gp_baseline.runGaussian("R" + str(run_count) + "_" + start_time, "Base_Initial")

        # Obtain Suggestions
        suggestions_X, suggestions_y = self.obtain_baseline_suggestions(run_count, start_time, gp_baseline, acq_func_obj,
                                                                      number_of_baseline_suggestions)

        PH.printme(PH.p1, number_of_baseline_suggestions, "Baseline suggestions:\n", suggestions_X.T, "\n", suggestions_y.T)
        gp_baseline.suggestions = {"suggestions_X": suggestions_X, "suggestions_y": suggestions_y}

        return gp_baseline

    def obtain_baseline_suggestions(self, run_count, start_time,  gp_baseline, acq_func_obj, number_of_baseline_suggestions):

        suggestions_X = []
        suggestions_y = []
        PH.printme(PH.p1, "Initial X and y:\n", gp_baseline.X, "\n", gp_baseline.y)

        for suggestion_count in range(number_of_baseline_suggestions):
            PH.printme(PH.p1, "Compute Suggestion: ", suggestion_count + 1)
            xnew, acq_func_values = acq_func_obj.max_acq_func(gp_baseline, suggestion_count + 1)
            xnew_orig = np.multiply(xnew.T, (gp_baseline.Xmax - gp_baseline.Xmin)) + gp_baseline.Xmin

            # Add the new observation point to the existing set of observed samples along with its true value
            X = gp_baseline.X
            X = np.append(X, [xnew], axis=0)

            ynew_orig = gp_baseline.fun_helper_obj.get_true_func_value(xnew_orig)
            ynew = (ynew_orig - gp_baseline.ymin) / (gp_baseline.ymax - gp_baseline.ymin)

            y = np.append(gp_baseline.y, [ynew], axis=0)

            # Normalising
            PH.printme(PH.p1, "(", xnew, ynew, ") is the new value added..    Original: ", (xnew_orig, ynew_orig))
            PH.printme(PH.p1, "Suggestion: ", suggestion_count + 1, " complete")
            gp_baseline.X = X
            gp_baseline.y = y

            suggestions_X.append(xnew)
            suggestions_y.append(ynew)

            # # Uncomment for debugging the suggestions and the posterior
            # PH.printme(PH.p1, "Final X and y:\n", gp_humanexpert.X, "\n", gp_humanexpert.y)

            with np.errstate(invalid='ignore'):
                mean, diag_variance, f_prior, f_post = gp_baseline.gaussian_predict(gp_baseline.Xs)
                standard_deviation = np.sqrt(diag_variance)
            gp_baseline.plot_posterior_predictions("R" + str(run_count) + "_" + start_time + "_Base_Suggestion" + "_" + str(
                suggestion_count+1),
                                                      gp_baseline.Xs,  gp_baseline.ys, mean, standard_deviation)

        suggestions_X = np.vstack(suggestions_X)
        suggestions_y = np.vstack(suggestions_y)
        return suggestions_X, suggestions_y
