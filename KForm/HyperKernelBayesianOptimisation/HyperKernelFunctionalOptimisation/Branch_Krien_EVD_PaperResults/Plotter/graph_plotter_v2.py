import os
import datetime
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# #report
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 22
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)

# #slides
# plt.rcParams["figure.figsize"] = (8, 6)
# plt.rcParams["font.size"] = 29
# plt.rc('xtick', labelsize=25)
# plt.rc('ytick', labelsize=25)

class GraphPlotter:

    def plotGraph(time_stamp):

        print("Graph Plotting at ", time_stamp)

        with open('file.txt', 'rt') as data_file:

            mean = ""
            f_err = ""
            X_obs = ""
            y_obs = ""
            ys_obs = ""

            start_mean_append = False
            start_err_append = False
            X_obs_append = False
            y_obs_append = False
            ys_obs_append = False
            mean_dict = {}
            err_dict ={}



            for myline in data_file:

                if(re.search("]", myline) and start_mean_append):
                    mean+=myline[:-2]
                    start_mean_append=False
                    mean_dict[mean_type] = mean
                    mean = ""
                    mean_type= ""

                if(re.search("]", myline) and start_err_append):
                    f_err+=myline[:-2]
                    start_err_append=False
                    err_dict[err_type] = f_err
                    f_err = ""
                    err_type = ""

                if (re.search("]]", myline) and X_obs_append):
                    X_obs += myline[2:-3]
                    X_obs_append =False

                if (re.search("]]", myline) and y_obs_append):
                    y_obs += myline[2:-3]
                    y_obs_append =False

                if (re.search("]]", myline) and ys_obs_append):
                    ys_obs += myline[2:-3]
                    ys_obs_append =False


                if(start_mean_append):
                    mean+=myline[:-1]
                    continue

                if (start_err_append):
                    f_err += myline[:-1]
                    continue

                if (X_obs_append):
                    X_obs += myline[2:-2]+" "
                    continue

                if (y_obs_append):
                    y_obs += myline[2:-2]+" "
                    continue

                if (ys_obs_append):
                    ys_obs += myline[2:-2]+" "
                    continue

                if (re.search('SE Mean'.lower(), myline.lower())):
                    mean+=myline[10:-1]
                    start_mean_append = True
                    mean_type = "SE"

                if (re.search('MATERN3 Mean'.lower(), myline.lower())):
                    mean+=myline[15:-1]
                    start_mean_append = True
                    mean_type = "MAT32"

                if (re.search('MKL Mean'.lower(), myline.lower())):
                    mean+=myline[11:-1]
                    start_mean_append = True
                    mean_type = "MKL"

                if (re.search('KFO Mean'.lower(), myline.lower())):
                    mean+=myline[11:-1]
                    start_mean_append = True
                    mean_type = "KFO"

                if (re.search('SE STD_DEV'.lower(), myline.lower())):
                    f_err += myline[13:-1]
                    start_err_append = True
                    err_type = "SE"

                if (re.search('MATERN3 STD_DEV'.lower(), myline.lower())):
                    f_err += myline[18:-1]
                    start_err_append = True
                    err_type = "MAT32"

                if (re.search('MKL STD_DEV'.lower(), myline.lower())):
                    f_err += myline[14:-1]
                    start_err_append = True
                    err_type = "MKL"

                if (re.search('KFO STD_DEV'.lower(), myline.lower())):
                    f_err += myline[14:-1]
                    start_err_append = True
                    err_type = "KFO"

                if (re.search('X'.lower(), myline.lower())):
                    X_obs += myline[5:-2]+" "
                    X_obs_append = True

                if (re.search('yy'.lower(), myline.lower())):
                    y_obs += myline[6:-2]+" "
                    y_obs_append = True

                if (re.search('ys'.lower(), myline.lower())):
                    ys_obs += myline[6:-2]+" "
                    ys_obs_append = True

            X_obs_array = np.array(re.split('[ ]+', X_obs))
            X_obs_array= X_obs_array.astype(np.float)
            y_obs_array = np.array(re.split('[ ]+', y_obs))
            y_obs_array= y_obs_array.astype(np.float)
            ys_obs_array = np.array(re.split('[ ]+', ys_obs))
            ys_obs_array= ys_obs_array.astype(np.float)

            # fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            #
            # x = np.arange(0, 5, 0.25)
            #
            # axes[0].plot(x, x ** 2, x, x ** 3)
            # axes[0].set_title("default axes ranges")
            #
            # axes[1].plot(x, x ** 2, x, x ** 3)
            # axes[1].axis('tight')
            # axes[1].set_title("tight axes")
            #
            # axes[2].plot(x, x ** 2, x, x ** 3)
            # axes[2].set_ylim([0, 60])
            # axes[2].set_xlim([2, 5])
            # axes[2].set_title("custom axes range")
            # fig.savefig("aa.pdf")
            # exit(0)


            final_mean = []
            final_err = []
            fun_type = ["SE", "MKL", "MAT32", "KFO"]
            # label_type = ["FIX", "ARD", "SVL", "MULTI"]
            label_type = ["SE", "MKL", "MAT32", "KFO"]

            for each_fun_type in fun_type:
                mean_array = np.array(re.split('[ ]+', mean_dict[each_fun_type]))
                err_array = np.array(re.split('[ ]+', err_dict[each_fun_type]))
                final_mean.append(mean_array.astype(np.float))
                final_err.append(err_array.astype(np.float))

            line_colors = ['red', 'blue', 'green', 'black']
            fill_colors = ['red', 'blue', 'green', 'grey']
            line_style = ['dashdot','dashed','dotted',(0,(3,1,1,1,1,1))]
            fig, ax = plt.subplots()

            iterations_axes_values = np.linspace(0, 1, 500)

            for j in np.arange(len(final_mean)):

                ax.plot(iterations_axes_values,final_mean[j],color=line_colors[j],ls=line_style[j], lw=4)
                plt.gca().fill_between(iterations_axes_values, final_mean[j]+ final_err[j],
                                       final_mean[j] - final_err[j], alpha=0.05, label=label_type[j],color=fill_colors[j])

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel('#Outfut f(x)')
            plt.ylabel('input')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            print(X_obs_array, y_obs_array)

            ax.plot(X_obs_array, y_obs_array, "bo")
            ax.plot(iterations_axes_values, ys_obs_array, "b", color="orange", lw=2)

            custom_lines = [Line2D([0],[0],  lw=3, ls="dashdot", color= 'red'),
                            Line2D([0],[0],  lw=3, ls="dashed",color='blue'),
                            Line2D([0], [0], lw=3, ls="dotted", color='green'),
                            Line2D([0],[0],  lw=3, ls=(0,(3,1,1,1,1,1)), color='black')
                            # Line2D([0], [0], lw=3, ls="dotted", color='black'),
                            # Line2D([0], [0], lw=3, ls="solid", color='green')
                            ]
            ax.legend(custom_lines, label_type, loc=1, fontsize='small')
            plt.title("Posterior Distribution")
            fig.savefig("sinc.pdf", pad_inches=0, bbox_inches='tight')
            # fig.savefig("sinc.eps", pad_inches=0, bbox_inches='tight')
            plt.autoscale(tight=True)
            plt.show()


    if __name__ == "__main__":
        timenow = datetime.datetime.now()
        stamp =  timenow.strftime("%H%M%S_%d%m%Y")
        # f = open('console_output_'+str(stamp)+'.txt', 'w')
        # original = sys.stdout
        # sys.stdout = Custom_Print(sys.stdout, f)
        plotGraph(timenow)








