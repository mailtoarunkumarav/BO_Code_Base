import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np
import datetime
import sys
import os
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
random_seed = 400
np.random.seed(random_seed)


# Class for starting Bayesian Optimization with the specified parameters
class ExpAIKerOptWrapper:

    def kernel_opt_wrapper(self, pwd_qualifier, full_time_stamp, function_type, external_input):

        number_of_runs = 7
        number_of_restarts_acq = 100
        number_of_minimiser_restarts = 100

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
        number_of_observations_groundtruth = 70
        number_of_random_observations_humanexpert = 3

        # Initial number of suggestions from human expert
        number_of_humanexpert_suggestions = 3
        number_of_suggestions_ai_baseline = 15

        epsilon_distance = 0.5

        noisy_suggestions = False

        plot_iterations = 0

        # acquisition type
        # acq_fun = 'ei'
        # acq_fun = 'ucb'

        acq_fun_list = ['ei', 'ucb']

        # total_regret_ai = []
        # # total_ei_regret_ai = []
        # total_regret_baseline = []
        # # total_ei_regret_baseline = []

        total_regret_ai = {}
        total_regret_baseline = {}

        PH.printme(PH.p1, "\n###################################################################\n Expert Full Obs, with ACQ: ",
                   acq_fun_list, "\n Number of Suggestions: ", number_of_suggestions_ai_baseline, "   Restarts: ",
                   number_of_minimiser_restarts)
        timenow = datetime.datetime.now()
        PH.printme(PH.p1, "Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for run_count in range(number_of_runs):

            PH.printme(PH.p1, "Constructing Kernel for ground truth....")

            gp_wrapper_obj = GPRegressorWrapper()

            gp_groundtruth = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "GroundTruth", number_of_observations_groundtruth,
                                                                function_type, None)
            gp_groundtruth.runGaussian(pwd_qualifier+"R" + str(run_count+1), "GT")
            PH.printme(PH.p1, "Ground truth kernel construction complete")

            acq_func_obj = AcquisitionFunction(None, number_of_restarts_acq, nu, epsilon1, epsilon2)

            for acq_fun in acq_fun_list:

                PH.printme(PH.p1, "\n\n########Generating results for Acquisition Function: ", acq_fun.upper(), "#############")
                file_identifier = pwd_qualifier + "R" + str(run_count+1) + "_" + acq_fun.upper()
                acq_func_obj.set_acq_func_type(acq_fun)

                human_expert_model_obj = HumanExpertModel()
                gp_humanexpert = human_expert_model_obj.initiate_humanexpert_model(file_identifier, gp_groundtruth,
                                                                                   gp_wrapper_obj, function_type, acq_func_obj,
                                                                                   number_of_random_observations_humanexpert,
                                                                                   number_of_humanexpert_suggestions, noisy_suggestions,
                                                                                   plot_iterations)

                aimodel_obj = AIModel(epsilon_distance, number_of_minimiser_restarts)
                gp_aimodel = aimodel_obj.initiate_aimodel(file_identifier, gp_wrapper_obj, gp_humanexpert, function_type,
                                                          acq_func_obj, number_of_random_observations_humanexpert,
                                                          number_of_suggestions_ai_baseline, noisy_suggestions, plot_iterations)

                gp_aimodel.runGaussian(pwd_qualifier+"R" + str(run_count + 1) + "_" + acq_fun.upper(), "AI_final")

                baseline_model_obj = BaselineModel()
                gp_baseline_model = baseline_model_obj.initiate_baseline_model(file_identifier, gp_wrapper_obj, gp_humanexpert,
                                                                               function_type, acq_func_obj,
                                                                               number_of_random_observations_humanexpert,
                                                                               number_of_suggestions_ai_baseline, noisy_suggestions,
                                                                               plot_iterations)

                gp_baseline_model.runGaussian(pwd_qualifier+"R" + str(run_count + 1) + "_" + acq_fun.upper(), "Base_final")

                true_max = gp_humanexpert.fun_helper_obj.get_true_max()
                true_max_norm = (true_max - gp_humanexpert.ymin) / (gp_humanexpert.ymax - gp_humanexpert.ymin)

                ai_regret = {}
                baseline_regret = {}

                for i in range(number_of_random_observations_humanexpert + number_of_suggestions_ai_baseline):

                    if acq_fun not in ai_regret or acq_fun not in baseline_regret:
                        ai_regret[acq_fun] = []
                        baseline_regret[acq_fun] = []

                    if i <= number_of_random_observations_humanexpert - 1:
                        ai_regret[acq_fun].append(true_max_norm - np.max(gp_aimodel.y[0:number_of_random_observations_humanexpert]))
                        baseline_regret[acq_fun].append(true_max_norm - np.max(gp_baseline_model.y[0:number_of_random_observations_humanexpert]))
                    else:
                        ai_regret[acq_fun].append(true_max_norm - np.max(gp_aimodel.y[0:i + 1]))
                        baseline_regret[acq_fun].append(true_max_norm - np.max(gp_baseline_model.y[0:i + 1]))

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

            PH.printme(PH.p1, "\n\n@@@@@@@@@@@@@@ Round ", str(run_count+1)+" complete @@@@@@@@@@@@@@@@\n\n")

        PH.printme(PH.p1, "\n", total_regret_ai, "\n", total_regret_baseline)
        # # # Plotting regret
        self.plot_regret(pwd_qualifier, acq_fun_list, total_regret_ai, total_regret_baseline, len(gp_aimodel.y))

        endtimenow = datetime.datetime.now()
        PH.printme(PH.p1, "\nEnd time: ", endtimenow.strftime("%H%M%S_%d%m%Y"))

        # plt.show()

    def plot_regret(self, pwd_name, acq_fun_list, total_regret_ai, total_regret_base, total_number_of_obs):

        iterations_axes_values = [i + 1 for i in np.arange(total_number_of_obs)]
        fig_name = 'Regret'
        plt.figure(str(fig_name))
        plt.clf()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # # # AI model

        colors = ["#713326", "#22642E", "#0D28C3", "#0FE4EB"]
        count = 0
        for acq in acq_fun_list:

            regret_ai = np.vstack(total_regret_ai[acq])
            regret_mean_ai = np.mean(regret_ai, axis=0)
            regret_std_dev_ai = np.std(regret_ai, axis=0)
            PH.printme(PH.p1, "\nAI Regret Details\nTotal Regret:", acq.upper(), "\n", regret_ai, "\n\n", acq.upper(), " Regret Mean",
                       regret_mean_ai, "\n\n",acq.upper(), " Regret Deviation\n", regret_std_dev_ai)

            ax.plot(iterations_axes_values, regret_mean_ai, colors[count])
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, regret_mean_ai + regret_std_dev_ai,
                                   regret_mean_ai - regret_std_dev_ai, color=colors[count], alpha=0.25, label="AI-" + acq.upper())

            count += 1

            # Baseline model
            regret_base = np.vstack(total_regret_base[acq])
            regret_mean_base = np.mean(regret_base, axis=0)
            regret_std_dev_base = np.std(regret_base, axis=0)
            PH.printme(PH.p1,"\nBaseline Regret Details\nTotal Regret \n", regret_base, "\n\nRegret Mean", regret_mean_base,
                  "\n\nRegret Deviation\n", regret_std_dev_base)

            ax.plot(iterations_axes_values, regret_mean_base, colors[count])
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, regret_mean_base + regret_std_dev_base,
                                   regret_mean_base - regret_std_dev_base, color=colors[count], alpha=0.25,
                                   label="Baseline-"+acq.upper())

            count += 1

        plt.axis([1, len(iterations_axes_values), 0, 1])
        # plt.xticks(iterations_axes_values, iterations_axes_values)
        plt.title('Regret')
        plt.xlabel('Evaluations')
        plt.ylabel('Simple Regret')
        legend = ax.legend(loc=1, fontsize='x-small')
        plt.savefig(pwd_name+fig_name + '.pdf')

if __name__ == "__main__":
    timenow = datetime.datetime.now()
    timestamp = timenow.strftime("%H%M%S_%d%m%Y")
    input = None
    ker_opt_wrapper_obj = ExpAIKerOptWrapper()

    function_type = "OSC1D"
    # function_type = "BEN1D"
    # function_type = "GCL1D"
    # function_type = "ACK1D"

    full_time_stamp = function_type+"_"+timestamp
    directory_full_qualifier_name = os.getcwd()+"/../../Experimental_Results/ExpertKnowledgeResults/"+full_time_stamp+"/"
    PH(directory_full_qualifier_name)
    PH.printme(PH.p1, "Function Type: ", function_type)
    ker_opt_wrapper_obj.kernel_opt_wrapper(directory_full_qualifier_name, full_time_stamp, function_type, input)
