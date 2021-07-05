import matplotlib.pyplot as plt
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

    def kernel_opt_wrapper(self, start_time, input):

        number_of_runs = 3
        number_of_restarts_acq = 100
        number_of_minimiser_restarts = 100

        # Epsilons is the value used during the maximization of PI and EI ACQ functions
        # Greater value like Epsilon = 10 is more of exploration and Epsilon = 0.0001 (exploitation)
        # Epsilon1 used in PI : 3
        epsilon1 = 3
        # epsilon2 used in EI : 4
        epsilon2 = 4
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

        #Acquisistion type
        # acq_fun = 'ei'
        acq_fun = 'ucb'

        total_regret_ai = []
        total_regret_baseline = []

        PH.printme(PH.p1, "\n###################################################################\n Expert Last Obs only, with ACQ: ",
                   acq_fun,
                   "\n Number of Suggestions: ", number_of_suggestions_ai_baseline, "   Restarts: ", number_of_minimiser_restarts)
        timenow = datetime.datetime.now()
        PH.printme(PH.p1, "Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for run_count in range(number_of_runs):

            PH.printme(PH.p1, "Constructing Kernel for ground truth....")

            gp_wrapper_obj = GPRegressorWrapper()
            acq_func_obj = AcquisitionFunction(acq_fun, number_of_restarts_acq, nu, epsilon1, epsilon2)

            gp_groundtruth = gp_wrapper_obj.construct_gp_object(start_time, "GroundTruth", number_of_observations_groundtruth, None)
            gp_groundtruth.runGaussian("R" + str(run_count+1) + "_" + start_time, "GT")
            PH.printme(PH.p1, "Ground truth kernel construction complete")

            human_expert_model_obj = HumanExpertModel()
            gp_humanexpert = human_expert_model_obj.initiate_humanexpert_model(start_time, run_count+1, gp_groundtruth, gp_wrapper_obj,
                                                                               acq_func_obj,
                                                                               number_of_random_observations_humanexpert,
                                                                               number_of_humanexpert_suggestions)

            aimodel_obj = AIModel(epsilon_distance, number_of_minimiser_restarts)
            gp_aimodel = aimodel_obj.initiate_aimodel(start_time, run_count+1, gp_wrapper_obj, gp_humanexpert, acq_func_obj,
                                         number_of_random_observations_humanexpert, number_of_suggestions_ai_baseline)

            gp_aimodel.runGaussian("R" + str(run_count+1) + "_" + start_time, "AI_final")

            baseline_model_obj = BaselineModel()
            gp_baseline_model = baseline_model_obj.initiate_baseline_model(start_time, run_count+1, gp_wrapper_obj, gp_humanexpert,
                                                                           acq_func_obj,
                                                                           number_of_random_observations_humanexpert,
                                                                           number_of_suggestions_ai_baseline)

            gp_baseline_model.runGaussian("R" + str(run_count+1)+ "_" + + start_time, "Base_final")

            true_max = gp_humanexpert.fun_helper_obj.get_true_max()
            true_max_norm = (true_max - gp_humanexpert.ymin)/(gp_humanexpert.ymax - gp_humanexpert.ymin)

            ai_regret = []
            baseline_regret = []
            for i in range(number_of_random_observations_humanexpert + number_of_suggestions_ai_baseline):
                if i <= number_of_random_observations_humanexpert - 1:
                    ai_regret.append(true_max_norm - np.max(gp_aimodel.y[0:number_of_random_observations_humanexpert]))
                    baseline_regret.append(true_max_norm - np.max(gp_baseline_model.y[0:number_of_random_observations_humanexpert]))
                else:
                    ai_regret.append(true_max_norm - np.max(gp_aimodel.y[0:i + 1]))
                    baseline_regret.append(true_max_norm - np.max(gp_baseline_model.y[0:i + 1]))
            total_regret_ai.append(ai_regret)
            total_regret_baseline.append(baseline_regret)

            # # Dummy Data for AI model and Baseline runs
            # total_regret_ai.append(np.multiply(baseline_regret, 0.85))
            # total_regret_ai.append(np.multiply(baseline_regret, 0.75))
            # total_regret_baseline.append(baseline_regret)
            # total_regret_baseline.append(np.multiply(baseline_regret, 0.5))

            PH.printme(PH.p1, "\n\n@@@@@@@@@@@@@@ Round ", str(run_count+1)+" complete @@@@@@@@@@@@@@@@\n\n")

        # # # Plotting regret
        self.plot_regret(start_time, total_regret_ai, total_regret_baseline, len(gp_aimodel.y), acq_fun)

        endtimenow = datetime.datetime.now()
        PH.printme(PH.p1, "\nEnd time: ", endtimenow.strftime("%H%M%S_%d%m%Y"))

        # plt.show()

    def plot_regret(self, start_time, total_regret_ai, total_regret_base, total_number_of_obs, acq_fun):

        iterations_axes_values = [i + 1 for i in np.arange(total_number_of_obs)]
        fig_name = 'Regret_'+start_time
        plt.figure(str(fig_name))
        plt.clf()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # # # AI model
        regret_ai = np.vstack(total_regret_ai)
        regret_mean_ai = np.mean(regret_ai, axis=0)
        regret_std_dev_ai = np.std(regret_ai, axis=0)
        PH.printme(PH.p1,"\nAI Regret Details\nTotal Regret \n", regret_ai, "\n\nRegret Mean", regret_mean_ai,
              "\n\nRegret Deviation\n", regret_std_dev_ai)

        PH.printme(PH.p1, regret_mean_ai)

        ax.plot(iterations_axes_values, regret_mean_ai, 'b')
        # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
        plt.gca().fill_between(iterations_axes_values, regret_mean_ai + regret_std_dev_ai,
                               regret_mean_ai - regret_std_dev_ai, color="blue", alpha=0.25, label="AI-"+acq_fun.upper())

        # Baseline model
        regret_base = np.vstack(total_regret_base)
        regret_mean_base = np.mean(regret_base, axis=0)
        regret_std_dev_base = np.std(regret_base, axis=0)
        PH.printme(PH.p1,"\nBaseline Regret Details\nTotal Regret \n", regret_base, "\n\nRegret Mean", regret_mean_base,
              "\n\nRegret Deviation\n", regret_std_dev_base)

        ax.plot(iterations_axes_values, regret_mean_base, 'g')
        # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
        plt.gca().fill_between(iterations_axes_values, regret_mean_base + regret_std_dev_base,
                               regret_mean_base - regret_std_dev_base, color="green", alpha=0.25, label="Baseline-"+acq_fun.upper())

        plt.axis([1, len(iterations_axes_values), 0, 1])
        plt.title('Regret')
        plt.xlabel('Evaluations')
        plt.ylabel('Simple Regret')
        legend = ax.legend(loc=1, fontsize='x-small')
        plt.savefig(fig_name + '.pdf')


if __name__ == "__main__":
    timenow = datetime.datetime.now()
    stamp = timenow.strftime("%H%M%S_%d%m%Y")
    PH(os.getcwd())
    input = None
    ker_opt_wrapper_obj = ExpAIKerOptWrapper()

    function_type = "OSC"
    # function_type = "BEN"
    # function_type = "GCL"
    # function_type = "ACK"

    PH.printme(PH.p1, "Function Type: ", function_type)
    stamp = function_type+"_"+stamp

    ker_opt_wrapper_obj.kernel_opt_wrapper(stamp, input)


