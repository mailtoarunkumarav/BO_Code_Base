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

        with open('graph_data.txt', 'rt') as data_file:

            mean = ""
            f_err = ""
            total_mean = []
            total_err =[]
            start_mean_append = False
            start_err_append = False
            first = False

            for myline in data_file:

                if(myline.endswith("]\n") and start_mean_append):
                    mean+=myline[:-2]
                    start_mean_append=False
                    total_mean.append(mean)
                    mean = ""

                if(myline.endswith("]\n") and start_err_append):
                    f_err+=myline[:-2]
                    start_err_append=False
                    total_err.append(f_err)
                    f_err = ""

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

                if (re.search('ARD Regret Mean'.lower(), myline.lower())):
                    mean+=myline[17:-1]
                    start_mean_append = True

                if (re.search('VAR Regret Mean'.lower(), myline.lower())):
                    mean+=myline[17:-1]
                    start_mean_append = True

                if (re.search('Multi Regret Mean'.lower(), myline.lower())):
                    mean+=myline[19:-1]
                    start_mean_append = True

                if (re.search('FIX Regret Deviation_std_error'.lower(), myline.lower())):
                    start_err_append = True
                    first = True

                if (re.search('VAR Regret Deviation_std_error'.lower(), myline.lower())):
                    start_err_append = True
                    first = True

                if (re.search('ARD Regret Deviation_std_error'.lower(), myline.lower())):
                    start_err_append = True
                    first = True

                if (re.search('Multi Regret Deviation_std_error'.lower(), myline.lower())):
                    start_err_append = True
                    first = True


            final_mean = []
            final_err = []
            for i in np.arange(len(total_mean)):
                mean_array = np.array(re.split('[ ]+', total_mean[i]))
                err_array = np.array(re.split('[ ]+', total_err[i]))
                final_mean.append(mean_array.astype(np.float))
                final_err.append(err_array.astype(np.float))

            label= ["FIX", "ARD", "VAR", "MULTI"]
            line_colors = ['red', 'blue', 'green', 'black']
            fill_colors = ['red', 'blue', 'green', 'grey']
            line_style = ['dashdot','dashed','solid','dotted']
            fig, ax = plt.subplots()
            iterations_axes_values = np.arange(1, 24)

            for j in np.arange(len(final_mean)):

                ax.plot(iterations_axes_values,final_mean[j],color=line_colors[j],ls=line_style[j])
                plt.gca().fill_between(iterations_axes_values, final_mean[j]+ final_err[j],
                                       final_mean[j] - final_err[j], alpha=0.25, label=label[j],color=fill_colors[j])

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title('Accuracy of Elastic Net')
            plt.xlabel('#Evaluations')
            plt.ylabel('Accuracy')
            plt.xlim(1, len(iterations_axes_values))
            custom_lines = [Line2D([0],[0],  lw=2, ls="dashdot", color= 'red'),
                            Line2D([0],[0],  lw=2, ls="dashed",color='blue'),
                            Line2D([0], [0], lw=2, ls="solid", color='green'),
                            Line2D([0],[0],  lw=2, ls="dotted", color='black')
                            ]

            ax.legend(custom_lines, label,loc=4, fontsize='x-small')

            plt.show()



    if __name__ == "__main__":
        timenow = datetime.datetime.now()
        stamp =  timenow.strftime("%H%M%S_%d%m%Y")
        # f = open('console_output_'+str(stamp)+'.txt', 'w')
        # original = sys.stdout
        # sys.stdout = Custom_Print(sys.stdout, f)
        plotGraph(timenow)







