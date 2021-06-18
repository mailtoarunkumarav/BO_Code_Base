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

        number_of_runs = 1
        number_of_restarts_acq = 10
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
        number_of_observations_groundtruth = 50
        number_of_observations_humanexpert = 5

        # Initial number of suggestions from human expert
        number_of_humanexpert_suggestions = 3
        number_of_ai_suggestions = 5

        epsilon_distance = 0.5
        number_of_minimiser_restarts = 20

        #Acquisistion type
        # acq_fun = 'ei'
        acq_fun = 'ucb'

        total_ucb_regret_ai = []
        total_ucb_regret_baseline = []

        PH.printme(PH.p1, "\n###################################################################\n")
        timenow = datetime.datetime.now()
        PH.printme(PH.p1, "Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for run_count in range(number_of_runs):

            PH.printme(PH.p1, "Constructing Kernel for ground truth....")

            gp_wrapper_obj = GPRegressorWrapper()
            acq_func_obj = AcquisitionFunction(acq_fun, number_of_restarts_acq, nu, epsilon1, epsilon2)

            gp_groundtruth = gp_wrapper_obj.construct_gp_object(start_time, "GroundTruth", number_of_observations_groundtruth, None)
            gp_groundtruth.runGaussian("R" + str(run_count) + start_time, "GT")
            PH.printme(PH.p1, "Ground truth kernel construction complete")

            human_expert_model_obj = HumanExpertModel()
            gp_humanexpert = human_expert_model_obj.initiate_humanexpert_model(start_time, run_count, gp_groundtruth, gp_wrapper_obj,
                                                                               acq_func_obj, number_of_observations_humanexpert,
                                                                               number_of_humanexpert_suggestions)

            baseline_model_obj = BaselineModel()
            gp_baseline_model = baseline_model_obj.initiate_baseline_model(start_time, run_count, gp_wrapper_obj, gp_humanexpert,
                                                                               acq_func_obj, number_of_observations_humanexpert,
                                                                               number_of_humanexpert_suggestions)

            aimodel_obj = AIModel(epsilon_distance, number_of_minimiser_restarts)
            gp_aimodel = aimodel_obj.initiate_aimodel(start_time, run_count, gp_wrapper_obj, gp_humanexpert, acq_func_obj,
                                         number_of_observations_humanexpert, number_of_ai_suggestions)

            true_max = gp_humanexpert.fun_helper_obj.get_true_max()
            true_max_norm = (true_max - gp_humanexpert.ymin)/(gp_humanexpert.ymax - gp_humanexpert.ymin)

            ai_regret = []
            baseline_regret = []
            for i in range(number_of_observations_humanexpert+number_of_humanexpert_suggestions):
                if i <= number_of_observations_humanexpert - 1:
                    ai_regret.append(true_max_norm - np.max(gp_aimodel.y[0:number_of_observations_humanexpert]))
                    baseline_regret.append(true_max_norm - np.max(gp_baseline_model.y[0:number_of_observations_humanexpert]))
                else:
                    ai_regret.append(true_max_norm - np.max(gp_aimodel.y[0:i + 1]))
                    baseline_regret.append(true_max_norm - np.max(gp_baseline_model.y[0:i + 1]))
            total_ucb_regret_ai.append(ai_regret)
            total_ucb_regret_baseline.append(baseline_regret)

            # # Dummy Data for AI model and Baseline runs
            # total_ucb_regret_ai.append(np.multiply(baseline_regret, 0.85))
            # total_ucb_regret_ai.append(np.multiply(baseline_regret, 0.75))
            # total_ucb_regret_baseline.append(baseline_regret)
            # total_ucb_regret_baseline.append(np.multiply(baseline_regret, 0.5))

        # # # Plotting regret
        self.plot_regret(total_ucb_regret_ai, total_ucb_regret_baseline, len(gp_baseline_model.y))

        plt.show()

    def plot_regret(self, total_ucb_regret_ai, total_ucb_regret_base, total_number_of_obs):

        iterations_axes_values = [i + 1 for i in np.arange(total_number_of_obs)]
        fig_name = 'Regret_'
        plt.figure(str(fig_name))
        plt.clf()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # AI model
        ucb_regret_ai = np.vstack(total_ucb_regret_ai)
        ucb_regret_mean_ai = np.mean(ucb_regret_ai, axis=0)
        ucb_regret_std_dev_ai = np.std(ucb_regret_ai, axis=0)
        print("\nAI Regret Details\nTotal UCB Regret \n", ucb_regret_ai, "\n\nUCB Regret Mean", ucb_regret_mean_ai,
              "\n\nUCB Regret Deviation\n", ucb_regret_std_dev_ai)

        print(ucb_regret_mean_ai)

        ax.plot(iterations_axes_values, ucb_regret_mean_ai, 'b')
        # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
        plt.gca().fill_between(iterations_axes_values, ucb_regret_mean_ai + ucb_regret_std_dev_ai,
                               ucb_regret_mean_ai - ucb_regret_std_dev_ai, color="blue", alpha=0.25, label="AI UCB")

        # Baseline model
        ucb_regret_base = np.vstack(total_ucb_regret_base)
        ucb_regret_mean_base = np.mean(ucb_regret_base, axis=0)
        ucb_regret_std_dev_base = np.std(ucb_regret_base, axis=0)
        print("\nBaseline Regret Details\nTotal UCB Regret \n", ucb_regret_base, "\n\nUCB Regret Mean", ucb_regret_mean_base,
              "\n\nUCB Regret Deviation\n", ucb_regret_std_dev_base)

        ax.plot(iterations_axes_values, ucb_regret_mean_base, 'g')
        # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
        plt.gca().fill_between(iterations_axes_values, ucb_regret_mean_base + ucb_regret_std_dev_base,
                               ucb_regret_mean_base - ucb_regret_std_dev_base, color="green", alpha=0.25, label="Baseline UCB")

        plt.axis([1, len(iterations_axes_values), 0, 1])
        plt.title('Regret')
        plt.xlabel('Evaluations')
        plt.ylabel('Simple Regret')
        legend = ax.legend(loc=1, fontsize='x-small')


if __name__ == "__main__":
    timenow = datetime.datetime.now()
    stamp = timenow.strftime("%H%M%S_%d%m%Y")
    PH(os.getcwd())
    input = None
    ker_opt_wrapper_obj = ExpAIKerOptWrapper()
    ker_opt_wrapper_obj.kernel_opt_wrapper(stamp, input)
