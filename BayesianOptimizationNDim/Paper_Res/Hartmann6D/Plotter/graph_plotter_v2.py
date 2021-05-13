import os
import datetime
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

class GraphPlotter:

    def plotGraph(time_stamp):

        print("Graph Plotting at ", time_stamp)

        with open('hartmann6d.txt', 'rt') as data_file:

            mean = ""
            f_err = ""
            start_mean_append = False
            start_err_append = False
            first = False
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

                if(start_mean_append):
                    mean+=myline[:-1]
                    continue

                if (start_err_append):
                    if first:
                        f_err += myline[2:-1]
                        first = False
                    else:
                        f_err += myline[:-1]
                    continue

                if (re.search('FIX Regret Mean'.lower(), myline.lower())):
                    mean+=myline[17:-1]
                    start_mean_append = True
                    mean_type = "FIX"


                if (re.search('ARD Regret Mean'.lower(), myline.lower())):
                    mean+=myline[17:-1]
                    start_mean_append = True
                    mean_type = "ARD"

                if (re.search('VAR Regret Mean'.lower(), myline.lower())):
                    mean+=myline[17:-1]
                    start_mean_append = True
                    mean_type = "VAR"

                if (re.search('Multi Regret Mean'.lower(), myline.lower())):
                    mean+=myline[19:-1]
                    start_mean_append = True
                    mean_type = "MULTI"

                if (re.search('Regret Deviation_std_err'.lower(), myline.lower())):
                    start_err_append = True
                    first = True

                    if(myline.startswith("FIX")):
                        err_type = "FIX"
                    elif(myline.startswith("ARD")):
                        err_type = "ARD"
                    elif(myline.startswith("VAR")):
                        err_type = "VAR"
                    elif(myline.startswith("Multi")):
                        err_type = "MULTI"

            final_mean = []
            final_err = []
            fun_type = ["FIX", "ARD", "VAR", "MULTI"]
            for each_fun_type in fun_type:
                mean_array = np.array(re.split('[ ]+', mean_dict[each_fun_type]))
                err_array = np.array(re.split('[ ]+', err_dict[each_fun_type]))
                final_mean.append(mean_array.astype(np.float))
                final_err.append(err_array.astype(np.float))

            line_colors = ['red', 'blue', 'green', 'black']
            fill_colors = ['red', 'blue', 'green', 'grey']
            line_style = ['dashdot','dashed','solid','dotted']
            fig, ax = plt.subplots()

            dim = 6
            iter = 60
            iterations_axes_values = np.arange(1, iter+(dim+1)+1)

            for j in np.arange(len(final_mean)):

                ax.plot(iterations_axes_values,final_mean[j],color=line_colors[j],ls=line_style[j])
                plt.gca().fill_between(iterations_axes_values, final_mean[j]+ final_err[j],
                                       final_mean[j] - final_err[j], alpha=0.25, label=fun_type[j],color=fill_colors[j])

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title('Simple Regret Hartmann 6D Function')
            plt.xlabel('#Evaluations')
            plt.ylabel('Regret')
            plt.xlim(1, len(iterations_axes_values))
            custom_lines = [Line2D([0],[0],  lw=1, ls="dashdot", color= 'red'),
                            Line2D([0],[0],  lw=1, ls="dashed",color='blue'),
                            Line2D([0], [0], lw=1, ls="solid", color='green'),
                            Line2D([0],[0],  lw=1, ls="dotted", color='black')
                            ]

            ax.legend(custom_lines, fun_type,loc=1, fontsize='x-small')

            plt.show()



    if __name__ == "__main__":
        timenow = datetime.datetime.now()
        stamp =  timenow.strftime("%H%M%S_%d%m%Y")
        # f = open('console_output_'+str(stamp)+'.txt', 'w')
        # original = sys.stdout
        # sys.stdout = Custom_Print(sys.stdout, f)
        plotGraph(timenow)







