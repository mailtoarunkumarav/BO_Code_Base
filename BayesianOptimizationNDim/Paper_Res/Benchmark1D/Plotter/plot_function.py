import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 28
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)

#Benchmark Function
x = np.linspace(0,10, 1000)
y=  (np.exp(-x) * np.sin(2 * np.pi * x))

fig = plt.figure()
plt.plot(x,y, lw =4)
plt.xlabel('x')
plt.ylabel('output, f(x)')
plt.title("Benchmark 1D Function")
fig.savefig("ben1d_fun.pdf", pad_inches=0, bbox_inches='tight')
fig.savefig("ben1d_fun.eps", pad_inches=0, bbox_inches='tight')
plt.autoscale(tight=True)
plt.show()

