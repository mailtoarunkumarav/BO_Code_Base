import matplotlib

matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np
import datetime
import sys
import os
import random
from matplotlib.ticker import MaxNLocator

from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction
from HumanExpert_Model import HumanExpertModel
from Baseline_Model import BaselineModel
from AI_Model import AIModel

sys.path.append("..")

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
class ExpAIKerOptWrapper:

    def kernel_opt_wrapper(self, pwd_qualifier, full_time_stamp, function_type, external_input):

        number_of_runs = 6
        number_of_restarts_acq = 10
        number_of_minimiser_restarts = 10

        # Epsilons is the value used during the maximization of PI and EI ACQ functions
        # Greater value like Epsilon = 10 is more of exploration and Epsilon = 0.0001 (exploitation)
        # Epsilon1 used in PI : 3
        epsilon1 = 3
        # epsilon2 used in EI : 4
        epsilon2 = 0.01
        # Kappa value to be used during the maximization of UCB ACQ function, but this is overriden in the case if
        # Kappa is calculated at each iteration as a function of the iteration and other parameters
        # kappa=10 is more of exploration and kappa = 0.1 is more of exploitation
        nu = 0.1

        # Number of observations for human expert and ground truth models
        number_of_observations_groundtruth = 10
        number_of_random_observations_humanexpert = 3

        # Initial number of suggestions from human expert
        number_total_suggestions = 12

        epsilon_distance = 0.6

        noisy_suggestions = False

        plot_iterations = 5

        # acquisition type
        # acq_fun = 'ei'
        # acq_fun = 'ucb'

        # acq_fun_list = ['ei', 'ucb']
        acq_fun_list = ['ucb']

        # total_regret_ai = []
        # # total_ei_regret_ai = []
        # total_regret_baseline = []
        # # total_ei_regret_baseline = []

        total_regret_ai = {}
        total_regret_baseline = {}

        lambda_reg = 0.7
        lambda_mul = 10

        llk_threshold = 0.9

        PH.printme(PH.p1, "\n###################################################################",
                   "Acq. Functions:", acq_fun_list, "   Number of Suggestions:", number_total_suggestions, "   Minimiser Restarts:",
                   number_of_minimiser_restarts, "   Runs:", number_of_runs, "\nRestarts for Acq:", number_of_restarts_acq, "  Eps1:",
                   epsilon1, "   eps2:", epsilon2, "   No_obs_GT:", number_of_observations_groundtruth, "   Random Obs:",
                   number_of_random_observations_humanexpert, "\n   Total Suggestions: ", number_total_suggestions, "    Eps Dist.:",
                   epsilon_distance, "\nNoisy:", noisy_suggestions,
                   "   plot iterations:", plot_iterations, "   Lambda:", lambda_reg, "   lambda Multiplier:",lambda_mul,
                   "\n Threshold Value: ", llk_threshold,
                   "\nSpecial Inputs: Normalised Acquisition function values + Two stage maximisation")
        timenow = datetime.datetime.now()
        PH.printme(PH.p1, "Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for run_count in range(number_of_runs):

            gp_wrapper_obj = GPRegressorWrapper()
            human_expert_model_obj = HumanExpertModel()

            gp_humanexpert = human_expert_model_obj.construct_human_expert_model(run_count, pwd_qualifier,
                                                                                 number_of_observations_groundtruth,
                                                                                 function_type, gp_wrapper_obj,
                                                                                 number_of_random_observations_humanexpert,
                                                                                 noisy_suggestions)

            initial_random_observations_X = gp_humanexpert.X
            initial_random_observations_y = gp_humanexpert.y

            # # Commenting for the initial experiments
            # HE_input_iterations = np.sort(random.sample(range(number_of_random_observations_humanexpert+1,
            #                                           number_of_suggestions_ai_baseline-1), number_of_humanexpert_suggestions))

            # HE_input_iterations = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            # HE_input_iterations = [1, 2, 8, 9, 12, 13]
            HE_input_iterations = [2, 3, 7, 8]
            number_of_humanexpert_suggestions = len(HE_input_iterations)

            PH.printme(PH.p1, number_of_humanexpert_suggestions, " Human Expert Input Iterations: ", HE_input_iterations)

            acq_func_obj = AcquisitionFunction(None, number_of_restarts_acq, nu, epsilon1, epsilon2)

            for acq_fun in acq_fun_list:

                observations_pool_X = initial_random_observations_X
                observations_pool_y = initial_random_observations_y

                PH.printme(PH.p1, "\n\n########Generating results for Acquisition Function: ", acq_fun.upper(), "#############")
                plot_files_identifier = pwd_qualifier + "R" + str(run_count + 1) + "_" + acq_fun.upper()
                acq_func_obj.set_acq_func_type(acq_fun)

                he_suggestions_best = []
                he_suggestions_worst = []

                PH.printme(PH.p1, "Construct GP object for AI Model")
                gp_aimodel = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "ai", number_of_random_observations_humanexpert,
                                                                function_type, gp_humanexpert.initial_random_observations)

                gp_aimodel.HE_input_iterations = HE_input_iterations
                gp_aimodel.he_suggestions = None

                aimodel_obj = AIModel(epsilon_distance, number_of_minimiser_restarts, lambda_reg, lambda_mul, llk_threshold)

                PH.printme(PH.p1, "Construct GP object for baseline")
                gp_baseline = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "baseline", number_of_random_observations_humanexpert,
                                                                 function_type, gp_humanexpert.initial_random_observations)

                gp_baseline.runGaussian(plot_files_identifier, "Base_Initial", True)
                PH.printme(PH.p1, "*************GP Constructions complete************")
                baseline_model_obj = BaselineModel()

                for suggestion_count in range(1, number_total_suggestions+1):

                    if suggestion_count in HE_input_iterations:
                        PH.printme(PH.p1, "\n\nStarting suggestion:", suggestion_count, " with Human Expert inputs......\nGenerating "
                                                                                        "Human Expert Suggestions")
                        gp_humanexpert.gp_fit(observations_pool_X, observations_pool_y)

                        xnew_best, xnew_worst = human_expert_model_obj.obtain_human_expert_suggestions(suggestion_count,
                                                                                                       plot_files_identifier + "_HE",
                                                                                                       gp_humanexpert,
                                                                                                       acq_func_obj, noisy_suggestions,
                                                                                                       plot_iterations)

                        he_suggestions_best.append(xnew_best)
                        he_suggestions_worst.append(xnew_worst)

                        gp_aimodel.he_suggestions = {"x_suggestions_best": he_suggestions_best, "x_suggestions_worst": he_suggestions_worst}

                    else:
                        PH.printme(PH.p1, "\n\nPredicting suggestion:" + str(suggestion_count)+" without Human Expert inputs")

                    aimodel_obj.max_acq_difference = -1 * float("inf")
                    aimodel_obj.max_llk = -1 * float("inf")
                    aimodel_obj.min_acq_difference = 1 * float("inf")
                    aimodel_obj.min_llk = 1 * float("inf")

                    # xnew_ai = aimodel_obj.obtain_aimodel_suggestions(plot_files_identifier + "_AI_", gp_aimodel, acq_func_obj,
                    #                                                  noisy_suggestions, suggestion_count, plot_iterations)
                    xnew_ai = aimodel_obj.obtain_twostg_aimodel_suggestions(plot_files_identifier + "_AI_", gp_aimodel,
                                                                     acq_func_obj, noisy_suggestions, suggestion_count,
                                                                     plot_iterations)


                    PH.printme(PH.p1, "Distance optimisation details:\nMax Diff:", aimodel_obj.max_acq_difference, "\tMin Diff:",
                               aimodel_obj.min_acq_difference,"\nMax likelihood:", aimodel_obj.max_llk, "\tMin Likelihood: ",
                               aimodel_obj.min_llk)

                    xnew_ai_orig = np.multiply(xnew_ai.T, (gp_aimodel.Xmax - gp_aimodel.Xmin)) + gp_aimodel.Xmin

                    # Add the new observation point to the existing set of observed samples along with its true value
                    observations_pool_X = np.append(observations_pool_X, [xnew_ai], axis=0)
                    ynew_ai_orig = gp_aimodel.fun_helper_obj.get_true_func_value(xnew_ai_orig)
                    ynew_ai = (ynew_ai_orig - gp_aimodel.ymin) / (gp_aimodel.ymax - gp_aimodel.ymin)
                    observations_pool_y = np.append(observations_pool_y, [ynew_ai], axis=0)

                    gp_aimodel.gp_fit(observations_pool_X, observations_pool_y)

                    PH.printme(PH.p1, "AI: (", xnew_ai, ynew_ai, ") is the new best value added..    Original: ", (xnew_ai_orig,
                                                                                                                   ynew_ai_orig))

                    if gp_aimodel.number_of_dimensions == 1 and plot_iterations != 0 and suggestion_count % plot_iterations == 0:
                        with np.errstate(invalid='ignore'):
                            mean_ai, diag_variance_ai, f_prior_ai, f_post_ai = gp_aimodel.gaussian_predict(gp_aimodel.Xs)
                            standard_deviation_ai = np.sqrt(diag_variance_ai)

                        gp_aimodel.plot_posterior_predictions(plot_files_identifier + "_AI_suggestion" + "_" + str(suggestion_count),
                                                              gp_aimodel.Xs, gp_aimodel.ys, mean_ai, standard_deviation_ai)

                    # # # Baseline model
                    PH.printme(PH.p1, "\nSuggestion for Baseline at iteration", suggestion_count)
                    xnew_baseline, ynew_baseline = baseline_model_obj.obtain_baseline_suggestion(suggestion_count, plot_files_identifier,
                                                                                                 gp_baseline, acq_func_obj,
                                                                                                 noisy_suggestions,
                                                                                                 plot_iterations)
                    PH.printme(PH.p1, "Baseline: (", xnew_baseline, ynew_baseline, ") is the new value added")
                    # plt.show()

                gp_aimodel.runGaussian(pwd_qualifier + "R" + str(run_count + 1) + "_" + acq_fun.upper(), "AI_final", True)
                gp_baseline.runGaussian(pwd_qualifier + "R" + str(run_count + 1) + "_" + acq_fun.upper(), "Base_final", True)

                true_max = gp_humanexpert.fun_helper_obj.get_true_max()

                true_max_norm = (true_max - gp_humanexpert.ymin) / (gp_humanexpert.ymax - gp_humanexpert.ymin)

                ai_regret = {}
                baseline_regret = {}

                for i in range(number_of_random_observations_humanexpert + number_total_suggestions):

                    if acq_fun not in ai_regret or acq_fun not in baseline_regret:
                        ai_regret[acq_fun] = []
                        baseline_regret[acq_fun] = []

                    if i <= number_of_random_observations_humanexpert - 1:
                        ai_regret[acq_fun].append(true_max_norm - np.max(gp_aimodel.y[0:number_of_random_observations_humanexpert]))
                        baseline_regret[acq_fun].append(true_max_norm - np.max(gp_baseline.y[0:number_of_random_observations_humanexpert]))
                    else:
                        ai_regret[acq_fun].append(true_max_norm - np.max(gp_aimodel.y[0:i + 1]))
                        baseline_regret[acq_fun].append(true_max_norm - np.max(gp_baseline.y[0:i + 1]))

                if acq_fun not in total_regret_ai or acq_fun not in total_regret_baseline:
                    total_regret_ai[acq_fun] = []
                    total_regret_baseline[acq_fun] = []

                total_regret_ai[acq_fun].append(ai_regret[acq_fun])
                total_regret_baseline[acq_fun].append(baseline_regret[acq_fun])

                # # Dummy Data for AI model and Baseline runs
                # total_regret_ai.append(np.multiply(baseline_regret, 0.85))
                # total_regret_ai.append(np.multiply(baseline_regret, 0.75))
                # total_regret_baseline.append(baseline_regret)
                # total_regret_baseline.append(np.multiply(baseline_regret, 0.5))

            PH.printme(PH.p1, "\n###########\nTotal AI Regret:\n", total_regret_ai, "\nTotal Baseline Regret:\n",total_regret_baseline,
                       "\n###################")
            PH.printme(PH.p1, "\n\n@@@@@@@@@@@@@@ Round ", str(run_count + 1) + " complete @@@@@@@@@@@@@@@@\n\n")

        PH.printme(PH.p1, "Tot_AI:\n", total_regret_ai, "\n\nTot Base:\n", total_regret_baseline)
        # # # Plotting regret
        self.plot_regret(pwd_qualifier, full_time_stamp, acq_fun_list, total_regret_ai, total_regret_baseline, len(gp_aimodel.y))

        endtimenow = datetime.datetime.now()
        PH.printme(PH.p1, "\nEnd time: ", endtimenow.strftime("%H%M%S_%d%m%Y"))

        # plt.show()

    def plot_regret(self, pwd_name, full_time_stamp, acq_fun_list, total_regret_ai, total_regret_base, total_number_of_obs):

        iterations_axes_values = [i + 1 for i in np.arange(total_number_of_obs)]
        fig_name = 'Regret_'+full_time_stamp
        plt.figure(str(fig_name))
        plt.clf()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # # # AI model

        colors = ["#713326", "#22642E", "#0D28C3", "#EB0F0F"]
        count = 0
        for acq in acq_fun_list:
            regret_ai = np.vstack(total_regret_ai[acq])
            regret_mean_ai = np.mean(regret_ai, axis=0)
            regret_std_dev_ai = np.std(regret_ai, axis=0)
            PH.printme(PH.p1, "\nAI Regret Details\nTotal Regret:", acq.upper(), "\n", regret_ai, "\n\n", acq.upper(), " Regret Mean",
                       regret_mean_ai, "\n\n", acq.upper(), " Regret Deviation\n", regret_std_dev_ai)

            ax.plot(iterations_axes_values, regret_mean_ai, colors[count])
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, regret_mean_ai + regret_std_dev_ai,
                                   regret_mean_ai - regret_std_dev_ai, color=colors[count], alpha=0.25, label="AI-" + acq.upper())

            count += 1

            # Baseline model
            regret_base = np.vstack(total_regret_base[acq])
            regret_mean_base = np.mean(regret_base, axis=0)
            regret_std_dev_base = np.std(regret_base, axis=0)
            PH.printme(PH.p1, "\nBaseline Regret Details\nTotal Regret \n", regret_base, "\n\nRegret Mean", regret_mean_base,
                       "\n\nRegret Deviation\n", regret_std_dev_base)

            ax.plot(iterations_axes_values, regret_mean_base, colors[count])
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, regret_mean_base + regret_std_dev_base,
                                   regret_mean_base - regret_std_dev_base, color=colors[count], alpha=0.25,
                                   label="Baseline-" + acq.upper())

            count += 1

        plt.axis([1, len(iterations_axes_values), 0, 3*np.maximum(np.max(regret_mean_ai), np.max(regret_mean_base))])
        # plt.axis([1, len(iterations_axes_values), 0, 0.9])
        # plt.xticks(iterations_axes_values, iterations_axes_values)
        plt.title('Regret')
        plt.xlabel('Evaluations')
        plt.ylabel('Simple Regret')
        legend = ax.legend(loc=1, fontsize='x-small')
        plt.savefig(pwd_name + fig_name + '.pdf')


if __name__ == "__main__":
    timenow = datetime.datetime.now()
    timestamp = timenow.strftime("%H%M%S_%d%m%Y")
    input = None
    ker_opt_wrapper_obj = ExpAIKerOptWrapper()

    # function_type = "OSC1D"
    function_type = "BEN1D"
    # function_type = "GCL1D"
    # function_type = "ACK1D"
    # function_type = "OSC2D"
    # function_type = "PAR2D"

    full_time_stamp = function_type + "_" + timestamp
    directory_full_qualifier_name = os.getcwd() + "/../../Experimental_Results/ExpertKnowledgeResults/" + full_time_stamp + "/"
    PH(directory_full_qualifier_name)
    PH.printme(PH.p1, "Function Type: ", function_type)
    ker_opt_wrapper_obj.kernel_opt_wrapper(directory_full_qualifier_name, full_time_stamp, function_type, input)
