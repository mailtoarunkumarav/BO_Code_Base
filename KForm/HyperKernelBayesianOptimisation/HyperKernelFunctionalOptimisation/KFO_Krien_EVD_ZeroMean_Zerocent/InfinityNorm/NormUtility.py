import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import datetime
import sys
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["font.size"] = 38
plt.rc('xtick', labelsize=38)
plt.rc('ytick', labelsize=38)
import os
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH

import math
from sklearn.model_selection import train_test_split
import pandas as pd
# kernel_type = 0
# number_of_test_datapoints = 20
np.random.seed(500)
# noise = 0.0


class NormUtility:

    def inifinity_norm_calculator_csv_experiment1(self, filename1, filename2):

        PH.printme(PH.p1, "Reading File 1 : KFO kernel")
        with open(filename1) as file_name:
            array = np.loadtxt(file_name, delimiter=",")
        kernel_kfo = array.reshape(-1, 1)

        PH.printme(PH.p1, "Reading File 2 : Matern3/2 kernel")
        with open(filename2) as file_name:
            array = np.loadtxt(file_name, delimiter=",")
        kernel_matern = array.reshape(-1, 1)

        kernel_diff = kernel_kfo - kernel_matern

        inf_norm = np.linalg.norm(kernel_diff, ord=np.inf, axis=0)
        l2norm = np.linalg.norm(kernel_diff, ord=2, axis=0)

        PH.printme(PH.p1, "Experiment1: KFO vs Matern Kernel \nInfinity norm Difference: ", inf_norm, "\nL2-Norm Difference: ", l2norm)


if __name__ == "__main__":

    PH(os.getcwd())
    timenow = datetime.datetime.now()
    PH.printme(PH.p1, "\nStart time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    norm = NormUtility()
    norm.inifinity_norm_calculator_csv_experiment1("Kernel_KFO_Exp1.csv", "Kernel_Matern32_Exp1.csv")
    timenow = datetime.datetime.now()
    PH.printme(PH.p1, "\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    plt.show()

