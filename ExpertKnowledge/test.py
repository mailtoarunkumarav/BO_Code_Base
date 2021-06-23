import scipy.optimize as opt
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# setting up the global parameters for plotting graphs i.e, graph size and suppress warning from multiple graphs
# being plotted
plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams["font.size"] = 12
plt.rcParams['figure.max_open_warning'] = 0
# np.seterr(divide='ignore', invalid='ignore')

total_number_of_obs = 10

iterations_axes_values = [i + 1 for i in np.arange(total_number_of_obs)]
fig_name = 'Regret_'
plt.figure(str(fig_name))
plt.clf()
ax = plt.subplot(111)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# AI model

ucb_regret_ai = [[0.336, 0.336, 0.336, 0.235, 0.235, 0.235, 0.154, 0.154,  0.1, 0.1], [0.512, 0.512, 0.512, 0.205, 0.205, 0.205, 0.157,
                                                                                       0.157, 0.06, 0.05]]
ucb_regret_ai = np.vstack(ucb_regret_ai)
ucb_regret_mean_ai = np.mean(ucb_regret_ai, axis=0)
ucb_regret_std_dev_ai = np.std(ucb_regret_ai, axis=0)

ax.plot(iterations_axes_values, ucb_regret_mean_ai, 'b')
# plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
plt.gca().fill_between(iterations_axes_values, ucb_regret_mean_ai + ucb_regret_std_dev_ai,
                       ucb_regret_mean_ai - ucb_regret_std_dev_ai, color="blue", alpha=0.25, label="AI UCB")

# Baseline model

ucb_regret_base = [[0.336, 0.336, 0.336, 0.165, 0.160, 0.160, 0.112, 0.112,  0.07, 0.07], [0.512, 0.512, 0.512, 0.183, 0.183, 0.183, 0.183,
                                                                                      0.09,
                                                                                   0.09,
                                                                                   0.06]]
ucb_regret_base = np.vstack(ucb_regret_base)
ucb_regret_mean_base= np.mean(ucb_regret_base, axis=0)
ucb_regret_std_dev_base = np.std(ucb_regret_base, axis=0)

ucb_regret_mean_base = np.mean(ucb_regret_base, axis=0)
ucb_regret_std_dev_base = np.std(ucb_regret_base, axis=0)
print("\nBaseline Regret Details\nTotal UCB Regret \n", ucb_regret_base, "\n\nUCB Regret Mean", ucb_regret_mean_base,
           "\n\nUCB Regret Deviation\n", ucb_regret_std_dev_base)

ax.plot(iterations_axes_values, ucb_regret_mean_base, 'g')
# plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
plt.gca().fill_between(iterations_axes_values, ucb_regret_mean_base + ucb_regret_std_dev_base,
                       ucb_regret_mean_base - ucb_regret_std_dev_base, color="green", alpha=0.25, label="Baseline UCB")

plt.axis([1, len(iterations_axes_values), 0, 1])
plt.title('MKL')
plt.xlabel('Evaluations')
plt.ylabel('Simple Regret')
legend = ax.legend(loc=1, fontsize='x-small')
plt.show()
exit(0)

# Maximizers and Gradients

init_obs = 3
true_max = 1.05
obs = np.array([0.7, 0.2, 0.3, 0.15, 0.8, 0.2, 0.5, 0.85, 0.9, 0.95])
print(obs[0: 3])
regret = []
for i in range(len(obs)):
    if i <= init_obs-1:
        regret.append(true_max - np.max(obs[0:init_obs]))
    else:
        regret.append(true_max - np.max(obs[0:i+1]))

print(regret)
plt.plot(range(1, len(obs)+1), regret)
plt.show()
exit()






def myfun (a,b=10):
    print(a,b)


a = np.array([10,20,30,40])
b=np.array([5,15,25,35])

myfun(10)

exit()

def multistart(fun, x0min, x0max, N, full_output=False, args=()):
    max = None
    func_max = -1 * float('inf')
    a = [10,20]
    b = 20
    for i in range(N):
        # print("\n#########################")
        x0 = sp.random.uniform(x0min, x0max)
        res = opt.minimize(lambda x: -fun(x, a, b), x0
                                   # , jac= fun_grad
                                   , method='L-BFGS-B', bounds=[[x0min, x0max]]
                                   , options={
                                            'maxfun': 2000, 'maxiter': 2000
                                            # ,'disp':True
                                            }
                                   )

        if (res.success == False):
            print("Convergence failed, Skipping")
            continue

        val = -1 * res.fun
        # print("prev max:", func_max, "\tNew value: ", val, "\t at x=", res.x)
        if val > func_max:
            # print("New maximum found ")
            max = res
            func_max = val

    if max != None:
        print("max obtained is ", func_max
              # , " at ", max.x
              )

    return max


def fun(x, a ,b ):
    print(a,b)
    return (np.exp(-x) * np.sin(8 * np.pi * x)) + 1
    # return np.sin(x)
    # return - np.cos(x) + 0.01 * x ** 2 + 1
    # return ((np.exp(-x) * np.sin(3 * np.pi * x)) + 0.3)
    # return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)
    # return ( (np.sin(10* np.pi* x)/(2*x))+(x-1)**4)
    # w = 1 +((x-1)/4)
    # return  (np.sin(np.pi * w))**2 + ((w-1)**2)*(1+ (np.sin(2 * np.pi * w))**2)
    # return -np.sin(x) *(np.sin((x**2)/np.pi))**20

multistart(fun,0,1,10)