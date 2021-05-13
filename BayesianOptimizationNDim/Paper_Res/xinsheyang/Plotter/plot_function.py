import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 28
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)

# Gramacy and Lee function
x = np.linspace(-10,10,1000)
beta = 15
m = 5
y = -1 * (np.exp(-(x / beta) ** (2 * m)) - 2 * np.exp(-(x) ** 2) * (np.cos(x)) ** 2)

fig = plt.figure()
plt.plot(x,y, lw =4)
plt.xlabel('x')
plt.ylabel('output, f(x)')
plt.title("Xin-She Yang 1D Function")
fig.savefig("xin1d_fun.pdf", pad_inches=0, bbox_inches='tight')
fig.savefig("xin1d_fun.eps", pad_inches=0, bbox_inches='tight')
plt.autoscale(tight=True)
plt.show()

