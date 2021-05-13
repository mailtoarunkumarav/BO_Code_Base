import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 28
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)

# Gramacy and Lee function
x = np.linspace(0.5,2.5,1000)
y= -1 * ((((np.sin(10 * np.pi * x))/(2*(x)))) +(x-1) ** 4)

fig = plt.figure()
plt.plot(x,y, lw =4)
plt.xlabel('x')
plt.ylabel('output, f(x)')
plt.title("Gramacy Lee 1D Function")
fig.savefig("gclee1d_fun.pdf", pad_inches=0, bbox_inches='tight')
fig.savefig("gclee1d_fun.eps", pad_inches=0, bbox_inches='tight')
plt.autoscale(tight=True)
plt.show()

