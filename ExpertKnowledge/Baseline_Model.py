import numpy as np

from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

class BaselineModel:

    def obtain_baseline_suggestion(self, suggestion_count, plot_files_identifier, gp_baseline, acq_func_obj, noisy_suggestions,
                                   plot_iterations):

        print("Baseline weights  before generating suggestion: ", gp_baseline.len_weights)
        gp_baseline.runGaussian(plot_files_identifier + "_BaseSuggestion_" + str(suggestion_count), "baseline", False)
        xnew, acq_func_values = acq_func_obj.max_acq_func("baseline", noisy_suggestions, gp_baseline, suggestion_count)
        xnew_orig = np.multiply(xnew.T, (gp_baseline.Xmax - gp_baseline.Xmin)) + gp_baseline.Xmin

        # Add the new observation point to the existing set of observed samples along with its true value
        X = gp_baseline.X
        X = np.append(X, [xnew], axis=0)

        ynew_orig = gp_baseline.fun_helper_obj.get_true_func_value(xnew_orig)

        # objective function noisy
        if gp_baseline.fun_helper_obj.true_func_type == "LIN1D":
            ynew_orig = ynew_orig + np.random.normal(0, 0.01)

        ynew = (ynew_orig - gp_baseline.ymin) / (gp_baseline.ymax - gp_baseline.ymin)

        y = np.append(gp_baseline.y, [ynew], axis=0)

        # Normalising
        PH.printme(PH.p1, "(", xnew, ynew, ") is the new value added..    Original: ", (xnew_orig, ynew_orig))
        gp_baseline.gp_fit(X, y)

        # # Uncomment for debugging the suggestions and the posterior
        # PH.printme(PH.p1, "Final X and y:\n", gp_humanexpert.X, "\n", gp_humanexpert.y)

        # Uncomment to plot all iterations
        if gp_baseline.number_of_dimensions == 1 and plot_iterations != 0 and suggestion_count % plot_iterations == 0:
            with np.errstate(invalid='ignore'):
                mean, diag_variance, f_prior, f_post = gp_baseline.gaussian_predict(gp_baseline.Xs)
                standard_deviation = np.sqrt(diag_variance)
            gp_baseline.plot_posterior_predictions(plot_files_identifier + "_BaseSuggestion_" + str(suggestion_count), gp_baseline.Xs,
                                                   gp_baseline.ys, mean, standard_deviation)

        # PH.printme(PH.p1, "Weights before ending....", gp_baseline.len_weights)
        # gp_baseline.runGaussian(plot_files_identifier + "_BaseSuggestion_" + str(suggestion_count), "baseline", False)

        return xnew, ynew


